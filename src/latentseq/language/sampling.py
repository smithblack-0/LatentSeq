"""Language Sampling assigns latent-conditioned stochastic behavior to a CFG.

The module parses a CFG into source/sink choices, converts the configured sharpness control into a
combined-logit spread, scales that spread by `1/sqrt(num_cores)`, and samples one normalized raw
production distribution for every core and latent state.  Stochastic initialization consumes the
ambient NumPy RNG; this module does not seed, snapshot, or restore it.
"""

import functools
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import softmax
from scipy.stats import norm, qmc

from .cfg import ParsedCFG, parse_cfg
from .ir import LSPCFG


# helpers

PRIMARY_CONTROLS = {"pairwise_odds", "ppl", "nats"}


@functools.lru_cache(maxsize=64)
def _calibration_normals(num_transitions: int) -> np.ndarray:
    # Deterministic unscrambled Halton points approximate the Gaussian expectation without creating
    # or advancing any random stream. Clipping avoids infinite normal quantiles at exact endpoints.
    sample_count = 4096 if num_transitions <= 64 else 2048
    uniform = qmc.Halton(d=num_transitions, scramble=False).random(sample_count + 1)[1:]
    uniform = np.clip(uniform, 1e-12, 1.0 - 1e-12)
    return norm.ppf(uniform).astype(np.float64, copy=False)


def _row_entropy(probabilities: np.ndarray) -> np.ndarray:
    positive = probabilities > 0
    logs = np.zeros_like(probabilities)
    logs[positive] = np.log(probabilities[positive])
    return -(probabilities * logs).sum(axis=-1)


def _control_key(config: dict[str, float]) -> str:
    keys = PRIMARY_CONTROLS & set(config)
    if len(keys) != 1:
        raise ValueError("sampling control must contain exactly one primary control")
    return next(iter(keys))


def _validate_direct_control(control: dict[str, float], num_transitions: int) -> None:
    key = _control_key(control)
    value = control[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    if key == "pairwise_odds":
        if value < 1:
            raise ValueError("pairwise_odds must be >= 1")
    elif key == "ppl":
        if not 1 < value <= num_transitions:
            raise ValueError("ppl must lie in (1, num_transitions]")
    else:
        if not 0 < value <= math.log(num_transitions):
            raise ValueError("nats must lie in (0, log(num_transitions)]")


def _resolve_source_control(parsed: ParsedCFG, source_index: int) -> tuple[dict[str, float], bool]:
    source = parsed.source_nodes[source_index]
    if source in parsed.source_overrides:
        return dict(parsed.source_overrides[source]), False
    inherited = {"pairwise_odds": float(parsed.sampling_defaults["pairwise_odds"])}
    for key in ("min_ppl", "max_ppl"):
        if key in parsed.sampling_defaults:
            inherited[key] = float(parsed.sampling_defaults[key])
    return inherited, True


def _cap_inherited_temperature(
    temperature: float,
    config: dict[str, float],
    num_transitions: int,
) -> float:
    if "min_ppl" in config:
        min_ppl = config["min_ppl"]
        if min_ppl > num_transitions:
            raise ValueError("min_ppl exceeds the number of transitions for a source")
        maximum_temperature = get_pseudotemperature(
            {"ppl": min_ppl}, num_transitions
        )
        temperature = min(temperature, maximum_temperature)
    if "max_ppl" in config:
        max_ppl = config["max_ppl"]
        if max_ppl > num_transitions:
            # A bound above the source's maximum possible perplexity is inactive.
            max_ppl = float(num_transitions)
        minimum_temperature = get_pseudotemperature(
            {"ppl": max_ppl}, num_transitions
        )
        temperature = max(temperature, minimum_temperature)
    return temperature


def _initialize_distribution(
    valid_sinks: tuple[int, ...],
    sink_count: int,
    temperature: float,
    num_cores: int,
) -> np.ndarray:
    distribution = np.zeros(sink_count, dtype=np.float64)
    if len(valid_sinks) == 1:
        distribution[valid_sinks[0]] = 1.0
        return distribution
    core_std = temperature / math.sqrt(num_cores)
    logits = np.random.normal(0.0, core_std, size=len(valid_sinks))
    distribution[list(valid_sinks)] = softmax(logits)
    return distribution


# main


def expected_entropy(num_transitions: int, pseudo_temperature: float) -> float:
    """Estimate expected softmax entropy for Gaussian logits deterministically.

    Args:
        num_transitions: Number of legal outcomes, at least two.
        pseudo_temperature: Nonnegative standard deviation of the combined Gaussian logits.

    Returns:
        Deterministic numerical estimate in nats. No package or ambient RNG state is used.
    """
    if isinstance(num_transitions, bool) or not isinstance(num_transitions, int) or num_transitions < 2:
        raise ValueError("num_transitions must be an integer >= 2")
    if (
        isinstance(pseudo_temperature, bool)
        or not isinstance(pseudo_temperature, (int, float))
        or pseudo_temperature < 0
    ):
        raise ValueError("pseudo_temperature must be nonnegative")
    if pseudo_temperature == 0:
        return math.log(num_transitions)
    logits = _calibration_normals(num_transitions) * float(pseudo_temperature)
    probabilities = softmax(logits, axis=-1)
    return float(_row_entropy(probabilities).mean())


def get_pseudotemperature(
    sampling_config: dict[str, float],
    num_transitions: int,
) -> float:
    """Convert one direct sharpness control into combined-logit standard deviation.

    Args:
        sampling_config: Mapping containing exactly one of `pairwise_odds`, `ppl`, or `nats`.
        num_transitions: Number of legal expansions for the source.

    Returns:
        Nonnegative combined-logit spread before per-core compensation.
    """
    if isinstance(num_transitions, bool) or not isinstance(num_transitions, int) or num_transitions < 1:
        raise ValueError("num_transitions must be positive")
    if num_transitions == 1:
        return 0.0
    _validate_direct_control(sampling_config, num_transitions)
    key = _control_key(sampling_config)
    value = float(sampling_config[key])
    if key == "pairwise_odds":
        return math.log(value) / math.sqrt(2.0)

    target_entropy = math.log(value) if key == "ppl" else value
    if math.isclose(target_entropy, math.log(num_transitions), rel_tol=0.0, abs_tol=1e-14):
        return 0.0

    upper = 1.0
    while expected_entropy(num_transitions, upper) > target_entropy:
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("failed to bracket entropy calibration target")
    return float(
        brentq(
            lambda temperature: expected_entropy(num_transitions, temperature)
            - target_entropy,
            0.0,
            upper,
            xtol=1e-10,
            rtol=1e-10,
        )
    )


def sample_ls_pcfg(
    grammar: str,
    _parse_cfg=parse_cfg,
) -> LSPCFG:
    """Sample latent-conditioned PCFG probability tables for one CFG.

    Args:
        grammar: Human-readable CFG produced by `sample_cfg` or an equivalent valid CFG text.

    Returns:
        `LSPCFG` with float64 `[core, latent_state, source, sink]` probabilities. Invalid source/sink
        positions are exactly zero and each legal raw row is normalized.
    """
    parsed = _parse_cfg(grammar)
    num_cores = parsed.language_shape["num_cores"]
    max_num_states = parsed.language_shape["max_num_states"]
    source_count = len(parsed.source_nodes)
    sink_count = len(parsed.sink_nodes)
    probabilities = np.zeros(
        (num_cores, max_num_states, source_count, sink_count), dtype=np.float64
    )

    for source_index in range(source_count):
        valid_sinks = parsed.source_to_sinks[source_index]
        num_transitions = len(valid_sinks)
        if num_transitions == 0:
            raise ValueError("every CFG source must have at least one production")
        control, inherited = _resolve_source_control(parsed, source_index)
        direct = {key: value for key, value in control.items() if key in PRIMARY_CONTROLS}
        if num_transitions == 1:
            temperature = 0.0
        else:
            _validate_direct_control(direct, num_transitions)
            temperature = get_pseudotemperature(direct, num_transitions)
            if inherited:
                temperature = _cap_inherited_temperature(
                    temperature, control, num_transitions
                )

        for core_index in range(num_cores):
            for state_index in range(max_num_states):
                probabilities[core_index, state_index, source_index] = (
                    _initialize_distribution(
                        valid_sinks,
                        sink_count,
                        temperature,
                        num_cores,
                    )
                )

    return LSPCFG(
        grammar=grammar,
        probabilities=probabilities,
        source_nodes=dict(parsed.source_nodes),
        sink_nodes=dict(parsed.sink_nodes),
        source_to_sinks=dict(parsed.source_to_sinks),
        language_shape=dict(parsed.language_shape),
    )
