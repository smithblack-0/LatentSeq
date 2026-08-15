"""Language Analysis derives one shared representative model from an LS-PCFG.

Analysis first averages latent states within each core and then intersects those core averages.  All
terminal-depth, entropy, and recurrence diagnostics reuse the resulting offspring/emission operators
and one propagated expected-mass trace.  The result is passive and retains the construction needed
for later reporting without access to runtime trajectories.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from latentseq._validation import require_exact_keys, require_positive_int

from .ir import LSPCFG


# helpers


def _entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    positive = probabilities > 0
    logs = np.zeros_like(probabilities, dtype=np.float64)
    logs[positive] = np.log(probabilities[positive])
    return -(probabilities * logs).sum(axis=axis)


def _intersect_rows(core_rows: np.ndarray) -> np.ndarray:
    # core_rows [core, ..., sink]. Invalid sinks are zero in every contributing row.
    positive_everywhere = np.all(core_rows > 0, axis=0)
    log_product = np.zeros(core_rows.shape[1:], dtype=np.float64)
    if np.any(positive_everywhere):
        logs = np.zeros_like(core_rows, dtype=np.float64)
        positive = core_rows > 0
        logs[positive] = np.log(core_rows[positive])
        log_product = logs.sum(axis=0)
    result = np.zeros_like(log_product, dtype=np.float64)
    for source_index in range(result.shape[-2]):
        valid = positive_everywhere[source_index]
        if not np.any(valid):
            continue
        values = log_product[source_index, valid]
        values = values - values.max()
        weights = np.exp(values)
        result[source_index, valid] = weights / weights.sum()
    return result


def _build_operators(
    ls_pcfg: LSPCFG,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    source_count = len(ls_pcfg.source_nodes)
    source_index = {source: index for index, source in ls_pcfg.source_nodes.items()}
    terminal_ids = [
        symbol
        for rhs in ls_pcfg.sink_nodes.values()
        for symbol in rhs
        if isinstance(symbol, int)
    ]
    terminal_width = max(terminal_ids) + 1 if terminal_ids else 0
    offspring = np.zeros((source_count, source_count), dtype=np.float64)
    emission = np.zeros((source_count, terminal_width), dtype=np.float64)
    max_rhs_length = 0

    for source in range(source_count):
        for sink in ls_pcfg.source_to_sinks[source]:
            probability = reference[source, sink]
            rhs = ls_pcfg.sink_nodes[sink]
            max_rhs_length = max(max_rhs_length, len(rhs))
            for symbol in rhs:
                if isinstance(symbol, int):
                    emission[source, symbol] += probability
                else:
                    offspring[source, source_index[symbol]] += probability
    return offspring, emission, max_rhs_length


def _propagate(
    offspring: np.ndarray,
    emission: np.ndarray,
    trace_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_count = offspring.shape[0]
    source_mass = np.zeros((trace_depth + 1, source_count), dtype=np.float64)
    terminal_mass = np.zeros((trace_depth + 1, emission.shape[1]), dtype=np.float64)
    source_mass[0, 0] = 1.0
    for depth in range(trace_depth + 1):
        terminal_mass[depth] = source_mass[depth] @ emission
        if depth < trace_depth:
            source_mass[depth + 1] = source_mass[depth] @ offspring
    return source_mass, terminal_mass


def _weighted_by_depth(
    source_mass: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    # source_values may be [source] or [leading..., source]. Move source first for matrix multiply.
    active = source_mass.sum(axis=1)
    if source_values.ndim == 1:
        numerator = source_mass @ source_values
        return np.divide(
            numerator,
            active,
            out=np.zeros_like(numerator),
            where=active > 0,
        )
    flat = source_values.reshape((-1, source_values.shape[-1]))
    numerator = source_mass @ flat.T
    denominator = active[:, None]
    weighted = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return weighted.reshape((source_mass.shape[0],) + source_values.shape[:-1])


def _recursive_components(
    offspring: np.ndarray,
    source_mass: np.ndarray,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    adjacency = offspring > 0
    if adjacency.size == 0:
        return (), np.zeros((source_mass.shape[0], 0), dtype=np.float64)
    component_count, labels = connected_components(
        csr_matrix(adjacency), directed=True, connection="strong"
    )
    components: list[tuple[int, ...]] = []
    for label in range(component_count):
        members = tuple(sorted(np.flatnonzero(labels == label).tolist()))
        recursive = len(members) > 1 or (
            len(members) == 1 and bool(adjacency[members[0], members[0]])
        )
        if recursive:
            components.append(members)
    components.sort(key=lambda members: members[0])
    mass = np.zeros((source_mass.shape[0], len(components)), dtype=np.float64)
    for component_index, members in enumerate(components):
        mass[:, component_index] = source_mass[:, list(members)].sum(axis=1)
    return tuple(components), mass


# main


@dataclass(slots=True)
class LanguageAnalysis:
    """Store the representative language model and all diagnostics derived from its one trace."""

    trace_depth: int
    core_average_probabilities: np.ndarray
    reference_probabilities: np.ndarray
    nonterminal_offspring: np.ndarray
    terminal_emission: np.ndarray
    source_mass: np.ndarray
    terminal_mass: np.ndarray
    active_nonterminal_mass: np.ndarray
    total_terminal_mass: np.ndarray
    cumulative_terminal_mass: np.ndarray
    residual_active_mass: float
    normalized_terminal_depth_mass: np.ndarray | None
    reference_terminal_cdf: np.ndarray | None
    reference_terminal_depth_mean: float | None
    reference_terminal_depth_variance: float | None
    reference_entropy_source: np.ndarray
    reference_entropy_by_depth: np.ndarray
    state_entropy: np.ndarray
    core_average_entropy: np.ndarray
    within_state_entropy: np.ndarray
    hidden_latent_entropy: np.ndarray
    core_average_entropy_by_depth: np.ndarray
    within_state_entropy_by_depth: np.ndarray
    hidden_latent_entropy_by_depth: np.ndarray
    spectral_radius: float
    recursive_components: tuple[tuple[int, ...], ...]
    recursive_component_mass: np.ndarray
    max_rhs_length: int

    def summary(self) -> dict[str, object]:
        """Return a compact report assembled only from already-stored Analysis results."""
        return {
            "trace_depth": self.trace_depth,
            "spectral_radius": self.spectral_radius,
            "residual_active_mass": self.residual_active_mass,
            "terminal_depth_mean": self.reference_terminal_depth_mean,
            "terminal_depth_variance": self.reference_terminal_depth_variance,
            "max_rhs_length": self.max_rhs_length,
            "recursive_components": self.recursive_components,
        }


def analyze_language(
    ls_pcfg: LSPCFG,
    analysis_configuration: dict[str, object],
) -> LanguageAnalysis:
    """Build deterministic representative diagnostics for one LS-PCFG.

    Args:
        ls_pcfg: Semantic sampled language to analyze.
        analysis_configuration: Exact mapping containing positive integer `trace_depth`.

    Returns:
        Passive `LanguageAnalysis` retaining the representative model, shared trace, and derived
        statistics. No runtime sample state is accepted or constructed.
    """
    require_exact_keys(
        analysis_configuration, {"trace_depth"}, "Language Analysis configuration"
    )
    trace_depth = require_positive_int(
        analysis_configuration["trace_depth"], "trace_depth"
    )
    probabilities = ls_pcfg.probabilities
    if probabilities.ndim != 4 or probabilities.dtype != np.float64:
        raise ValueError("LS-PCFG probabilities must be float64 [core,state,source,sink]")

    core_average = probabilities.mean(axis=1)
    reference = _intersect_rows(core_average)
    offspring, emission, max_rhs_length = _build_operators(ls_pcfg, reference)
    source_mass, terminal_mass = _propagate(
        offspring, emission, trace_depth
    )

    active_nonterminal_mass = source_mass.sum(axis=1)
    total_terminal_mass = terminal_mass.sum(axis=1)
    cumulative_terminal_mass = np.cumsum(total_terminal_mass)
    residual_active_mass = float(active_nonterminal_mass[-1])
    represented_terminal_mass = float(total_terminal_mass.sum())
    if represented_terminal_mass > 0:
        normalized_depth = total_terminal_mass / represented_terminal_mass
        terminal_cdf = np.cumsum(normalized_depth)
        depths = np.arange(trace_depth + 1, dtype=np.float64)
        depth_mean = float(np.dot(depths, normalized_depth))
        depth_variance = float(
            np.dot((depths - depth_mean) ** 2, normalized_depth)
        )
    else:
        normalized_depth = None
        terminal_cdf = None
        depth_mean = None
        depth_variance = None

    reference_entropy_source = _entropy(reference)
    reference_entropy_by_depth = _weighted_by_depth(
        source_mass, reference_entropy_source
    )

    state_entropy = _entropy(probabilities)
    core_average_entropy = _entropy(core_average)
    within_state_entropy = state_entropy.mean(axis=1)
    hidden_latent_entropy = core_average_entropy - within_state_entropy
    core_average_entropy_by_depth = _weighted_by_depth(
        source_mass, core_average_entropy
    )
    within_state_entropy_by_depth = _weighted_by_depth(
        source_mass, within_state_entropy
    )
    hidden_latent_entropy_by_depth = _weighted_by_depth(
        source_mass, hidden_latent_entropy
    )

    eigenvalues = np.linalg.eigvals(offspring)
    spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    recursive_components, recursive_component_mass = _recursive_components(
        offspring, source_mass
    )

    return LanguageAnalysis(
        trace_depth=trace_depth,
        core_average_probabilities=core_average,
        reference_probabilities=reference,
        nonterminal_offspring=offspring,
        terminal_emission=emission,
        source_mass=source_mass,
        terminal_mass=terminal_mass,
        active_nonterminal_mass=active_nonterminal_mass,
        total_terminal_mass=total_terminal_mass,
        cumulative_terminal_mass=cumulative_terminal_mass,
        residual_active_mass=residual_active_mass,
        normalized_terminal_depth_mass=normalized_depth,
        reference_terminal_cdf=terminal_cdf,
        reference_terminal_depth_mean=depth_mean,
        reference_terminal_depth_variance=depth_variance,
        reference_entropy_source=reference_entropy_source,
        reference_entropy_by_depth=reference_entropy_by_depth,
        state_entropy=state_entropy,
        core_average_entropy=core_average_entropy,
        within_state_entropy=within_state_entropy,
        hidden_latent_entropy=hidden_latent_entropy,
        core_average_entropy_by_depth=core_average_entropy_by_depth,
        within_state_entropy_by_depth=within_state_entropy_by_depth,
        hidden_latent_entropy_by_depth=hidden_latent_entropy_by_depth,
        spectral_radius=spectral_radius,
        recursive_components=recursive_components,
        recursive_component_mass=recursive_component_mass,
        max_rhs_length=max_rhs_length,
    )
