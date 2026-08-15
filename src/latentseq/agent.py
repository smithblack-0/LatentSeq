"""Agent implements the latent-process half of LatentSeq.

A Core owns one reusable sparse Markov transition system and one dwell policy.  Agent only orders
Cores and stacks their records into the Latent Sample IR `[batch, core, timestep]`.  Construction is
owned by factory functions so class instances remain dependency-injectable and persistence restores
sampled machinery rather than rebuilding it.
"""

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch

from ._validation import require_exact_keys, require_positive_int
from ._sampling_ops import (
    build_configured_sampler,
    call_torch_sampler,
    sample_categorical,
    sample_uniform_integers,
)
from .persistence import (
    ensure_directory,
    load_npz,
    numpy_to_tensor,
    read_json,
    save_npz,
    tensor_to_numpy,
    write_json,
)


# helpers

CORE_FIELDS = {"hidden_size", "num_connections", "advancement_period", "concentration"}
AGENT_FIELDS = {"cores", "defaults"}
INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


def _validate_core_config(config: Mapping[str, object]) -> None:
    require_exact_keys(config, CORE_FIELDS, "Core configuration")
    hidden_size = config["hidden_size"]
    connections = config["num_connections"]
    concentration = config["concentration"]
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 2:
        raise ValueError("hidden_size must be an integer >= 2")
    if (
        isinstance(connections, bool)
        or not isinstance(connections, int)
        or not 2 <= connections <= hidden_size
    ):
        raise ValueError("num_connections must satisfy 2 <= num_connections <= hidden_size")
    if (
        isinstance(concentration, bool)
        or not isinstance(concentration, (int, float))
        or concentration <= 0
    ):
        raise ValueError("concentration must be positive")


def resolve_dwelltime_sampler(
    advancement_period: int | dict[str, object],
    device: torch.device,
    _probe: bool = True,
) -> Callable[[int], torch.Tensor]:
    """Resolve one dwell policy into a batch-width Torch sampler.

    Args:
        advancement_period: Nonnegative constant dwell or a mapping containing `function` and
            arguments for a Torch sampling function.
        device: Device on which returned dwell tensors must live.

    Returns:
        Callback accepting a positive batch width and returning nonnegative int64 `[batch]`.
    """
    if isinstance(advancement_period, int) and not isinstance(advancement_period, bool):
        if advancement_period < 0:
            raise ValueError("advancement_period cannot be negative")

        def constant_sampler(batch_width: int) -> torch.Tensor:
            return torch.full(
                (batch_width,), advancement_period, dtype=torch.int64, device=device
            )

        return constant_sampler

    if not isinstance(advancement_period, dict) or "function" not in advancement_period:
        raise ValueError(
            "advancement_period must be a nonnegative integer or Torch sampler mapping"
        )
    function_name = advancement_period["function"]
    if not isinstance(function_name, str) or not hasattr(torch, function_name):
        raise ValueError(f"unknown Torch dwell sampler {function_name!r}")
    function = getattr(torch, function_name)
    params = {key: value for key, value in advancement_period.items() if key != "function"}

    # New construction probes malformed samplers before generation. Pretrained restoration skips
    # that stochastic probe because the saved policy was already validated when constructed.
    if _probe:
        values = call_torch_sampler(function, params, 3, device)
        if values.shape != (3,):
            raise ValueError("dwell sampler must return exactly [batch]")
        if values.dtype not in INTEGER_DTYPES:
            raise ValueError("dwell sampler must return integer-valued tensors")
        if bool(torch.any(values < 0)):
            raise ValueError("dwell sampler returned negative values")

    # The opaque operator keeps the configured eager Torch sampling primitive intact under
    # `torch.compile`; it does not own or alter the ambient RNG stream.
    return build_configured_sampler(
        function_name,
        params,
        torch.empty(0, device=device),
    )


def _build_transition_table(
    hidden_size: int,
    num_connections: int,
    concentration: float,
    device: torch.device,
) -> torch.Tensor:
    # A random Hamiltonian cycle guarantees strong connectivity independently of the extra edges.
    permutation = torch.randperm(hidden_size, device=device)
    cycle_successor = torch.empty(hidden_size, dtype=torch.int64, device=device)
    cycle_successor[permutation] = torch.roll(permutation, shifts=-1)

    targets = torch.empty(
        (hidden_size, num_connections), dtype=torch.int64, device=device
    )
    targets[:, 0] = cycle_successor
    all_states = torch.arange(hidden_size, device=device)
    for state in range(hidden_size):
        candidates = all_states[all_states != cycle_successor[state]]
        order = torch.randperm(hidden_size - 1, device=device)
        targets[state, 1:] = candidates[order[: num_connections - 1]]

    alpha = torch.full(
        (hidden_size, num_connections),
        float(concentration),
        dtype=torch.float32,
        device=device,
    )
    gamma = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
    weights = gamma / gamma.sum(dim=1, keepdim=True)
    table = torch.zeros(
        (hidden_size, hidden_size), dtype=torch.float32, device=device
    )
    table.scatter_(1, targets, weights)
    return table


def _validate_initial_states(
    states: torch.Tensor,
    batch_width: int,
    num_states: int,
    device: torch.device,
) -> None:
    if (
        states.shape != (batch_width,)
        or states.dtype != torch.int64
        or states.device != device
    ):
        raise ValueError("initial_markov_states must be int64 [batch] on the Core device")
    if bool(torch.any(states < 0)) or bool(torch.any(states >= num_states)):
        raise ValueError("initial_markov_states contains an out-of-range state")


# main


class Core:
    """Provide one reusable latent Markov process with an independent dwell policy.

    Core stores only sampled static machinery. `draw_samples` creates Markov and dwell state for one
    call, advances the full batch with fixed-shape tensor operations, and returns the Core record.
    """

    def __init__(
        self,
        num_states: int,
        transition_table: torch.Tensor,
        advancement_period: int | dict[str, object],
        dwell_sampler: Callable[[int], torch.Tensor],
        construction_config: dict[str, object],
        device: torch.device,
    ) -> None:
        self.num_states = num_states
        self.transition_table = transition_table
        self.advancement_period = advancement_period
        self.dwell_sampler = dwell_sampler
        self.construction_config = construction_config
        self.device = device
        self.sample_backend = (
            torch.compile(self._draw_samples_unchecked, dynamic=False, fullgraph=True)
            if device.type == "cuda"
            else self._draw_samples_unchecked
        )

    def sample_dwell_times(self, batch_width: int) -> torch.Tensor:
        """Sample fresh dwell counts for the complete batch."""
        return self.dwell_sampler(batch_width)

    def sample_initial_markov_states(self, batch_width: int) -> torch.Tensor:
        """Sample uniform initial states for the complete batch."""
        return sample_uniform_integers(
            self.transition_table, 0, self.num_states, batch_width
        )

    def sample_transitions(self, markov_states: torch.Tensor) -> torch.Tensor:
        """Sample one unconditional successor from each selected transition row."""
        rows = self.transition_table[markov_states]
        return sample_categorical(rows)

    @torch.no_grad()
    def draw_samples(
        self,
        batch_width: int,
        num_timesteps: int,
        initial_markov_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate one `[batch, timestep]` Core record.

        Args:
            batch_width: Positive number of independent chains.
            num_timesteps: Positive number of states to emit.
            initial_markov_states: Optional exact int64 `[batch]` starting states. Dwell phase is
                freshly sampled even when states are supplied.

        Returns:
            int64 tensor on the Core device. The supplied starting state, when present, is emitted
            at timestep zero.
        """
        require_positive_int(batch_width, "batch_width")
        require_positive_int(num_timesteps, "num_timesteps")
        if initial_markov_states is not None:
            _validate_initial_states(
                initial_markov_states, batch_width, self.num_states, self.device
            )
        return self._draw_samples_unchecked(
            batch_width, num_timesteps, initial_markov_states
        )

    @torch.no_grad()
    def _draw_samples_unchecked(
        self,
        batch_width: int,
        num_timesteps: int,
        initial_markov_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Execute the fixed-shape sampling path after public argument validation."""
        if initial_markov_states is None:
            markov_states = self.sample_initial_markov_states(batch_width)
        else:
            markov_states = initial_markov_states

        dwell_times = self.sample_dwell_times(batch_width)
        record = torch.empty(
            (batch_width, num_timesteps), dtype=torch.int64, device=self.device
        )
        record[:, 0] = markov_states
        for timestep in range(1, num_timesteps):
            # Both stochastic branches execute for every lane; masking selects which result matters.
            transitioned_states = self.sample_transitions(markov_states)
            refreshed_dwell = self.sample_dwell_times(batch_width)
            should_transition = dwell_times == 0
            markov_states = torch.where(
                should_transition, transitioned_states, markov_states
            )
            dwell_times = torch.where(
                should_transition, refreshed_dwell, dwell_times - 1
            )
            record[:, timestep] = markov_states
        return record


class Agent:
    """Expose an ordered collection of latent Cores as one reusable latent-process object.

    Agent exists so callers can construct, pool, persist, and sample latent processes independently
    of any Language or Vocabulary. It contains no grammar or lexical behavior.
    """

    def __init__(self, cores: Sequence[Core], device: torch.device) -> None:
        self.cores = tuple(cores)
        self.device = device

    @property
    def state_signature(self) -> tuple[int, ...]:
        """Return each Core's latent-state width in compatibility order."""
        return tuple(core.num_states for core in self.cores)

    @classmethod
    def from_config(
        cls,
        config: dict[str, object],
        device: str | torch.device,
    ) -> "Agent":
        """Construct a newly sampled Agent from explicit configuration."""
        return build_agent_from_config(config, torch.device(device), _agent_cls=cls)

    @classmethod
    def from_pretrained(
        cls,
        path: str | os.PathLike[str],
        device: str | torch.device,
    ) -> "Agent":
        """Restore an exactly saved Agent on the selected execution device."""
        return load_agent_pretrained(path, torch.device(device), _agent_cls=cls)

    def save_pretrained(self, path: str | os.PathLike[str]) -> None:
        """Persist sampled reusable Agent state, excluding RNG and trajectory state."""
        save_agent_pretrained(self, path)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        length: int,
        initial_markov_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate the Latent Sample IR `[batch, core, timestep]`.

        Args:
            batch_size: Positive number of independent streams.
            length: Positive number of emitted positions.
            initial_markov_states: Optional int64 `[batch, core]` starting states.

        Returns:
            int64 latent sample on the Agent device.
        """
        require_positive_int(batch_size, "batch_size")
        require_positive_int(length, "length")
        if initial_markov_states is not None:
            expected = (batch_size, len(self.cores))
            if (
                initial_markov_states.shape != expected
                or initial_markov_states.dtype != torch.int64
                or initial_markov_states.device != self.device
            ):
                raise ValueError(
                    f"initial_markov_states must be int64 {expected} on the Agent device"
                )
            for core_index, core in enumerate(self.cores):
                states = initial_markov_states[:, core_index]
                if bool(torch.any(states < 0)) or bool(torch.any(states >= core.num_states)):
                    raise ValueError(
                        f"initial_markov_states contains an out-of-range state for Core {core_index}"
                    )
        records = []
        for core_index, core in enumerate(self.cores):
            initial = (
                None
                if initial_markov_states is None
                else initial_markov_states[:, core_index]
            )
            records.append(core.sample_backend(batch_size, length, initial))
        return torch.stack(records, dim=1)

    __call__ = sample


# construction


def build_core(
    config: dict[str, object],
    device: torch.device,
    _core_cls: type[Core] = Core,
    _resolve_dwelltime_sampler: Callable[
        [int | dict[str, object], torch.device], Callable[[int], torch.Tensor]
    ] = resolve_dwelltime_sampler,
) -> Core:
    """Construct one newly sampled Core from an exact configuration mapping."""
    _validate_core_config(config)
    construction_config = dict(config)
    dwell_sampler = _resolve_dwelltime_sampler(
        construction_config["advancement_period"], device
    )
    transition_table = _build_transition_table(
        int(construction_config["hidden_size"]),
        int(construction_config["num_connections"]),
        float(construction_config["concentration"]),
        device,
    )
    return _core_cls(
        num_states=int(construction_config["hidden_size"]),
        transition_table=transition_table,
        advancement_period=construction_config["advancement_period"],
        dwell_sampler=dwell_sampler,
        construction_config=construction_config,
        device=device,
    )


def build_agent(
    core_specifications: Sequence[dict[str, object]],
    defaults: dict[str, object],
    device: torch.device,
    _agent_cls: type[Agent] = Agent,
    _build_core: Callable[[dict[str, object], torch.device], Core] = build_core,
) -> Agent:
    """Build an Agent by merging explicit per-Core values over explicit caller defaults."""
    if (
        not isinstance(core_specifications, Sequence)
        or isinstance(core_specifications, (str, bytes))
        or not core_specifications
    ):
        raise ValueError("cores must be a nonempty sequence of mappings")
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping")

    cores: list[Core] = []
    for specification in core_specifications:
        if not isinstance(specification, dict):
            raise ValueError("every Core specification must be a mapping")
        unknown = (set(defaults) | set(specification)) - CORE_FIELDS
        if unknown:
            raise ValueError(f"unknown Core configuration fields: {sorted(unknown)}")
        merged = {**defaults, **specification}
        missing = CORE_FIELDS - set(merged)
        if missing:
            raise ValueError(f"missing Core configuration fields: {sorted(missing)}")
        cores.append(_build_core(merged, device))
    return _agent_cls(cores, device)


def build_agent_from_config(
    config: dict[str, object],
    device: torch.device,
    _agent_cls: type[Agent] = Agent,
    _build_agent: Callable[..., Agent] = build_agent,
) -> Agent:
    """Construct the public Agent from its exact top-level configuration."""
    require_exact_keys(config, AGENT_FIELDS, "Agent configuration")
    cores = config["cores"]
    defaults = config["defaults"]
    if not isinstance(cores, Sequence) or isinstance(cores, (str, bytes)):
        raise ValueError("Agent cores must be a sequence")
    if not isinstance(defaults, dict):
        raise ValueError("Agent defaults must be a mapping")
    return _build_agent(cores, defaults, device, _agent_cls=_agent_cls)


def save_agent_pretrained(agent: Agent, path: str | os.PathLike[str]) -> None:
    """Persist the exact static Core structures that define an Agent."""
    directory = ensure_directory(path)
    write_json(
        directory / "config.json",
        {
            "format": "latentseq-agent-v1",
            "cores": [core.construction_config for core in agent.cores],
        },
    )
    save_npz(
        directory / "state.npz",
        **{
            f"transition_{index}": tensor_to_numpy(core.transition_table)
            for index, core in enumerate(agent.cores)
        },
    )


def load_agent_pretrained(
    path: str | os.PathLike[str],
    device: torch.device,
    _agent_cls: type[Agent] = Agent,
    _core_cls: type[Core] = Core,
    _resolve_dwelltime_sampler: Callable[
        [int | dict[str, object], torch.device], Callable[[int], torch.Tensor]
    ] = resolve_dwelltime_sampler,
) -> Agent:
    """Restore saved Core tables directly without stochastic reconstruction."""
    directory = Path(path)
    manifest = read_json(directory / "config.json")
    require_exact_keys(manifest, {"format", "cores"}, "Agent pretrained manifest")
    if manifest["format"] != "latentseq-agent-v1":
        raise ValueError("unsupported Agent pretrained format")
    core_configs = manifest["cores"]
    if not isinstance(core_configs, list) or not core_configs:
        raise ValueError("Agent pretrained manifest requires a nonempty Core list")
    arrays = load_npz(directory / "state.npz")

    cores: list[Core] = []
    for index, raw_config in enumerate(core_configs):
        if not isinstance(raw_config, dict):
            raise ValueError("saved Core configuration must be a mapping")
        _validate_core_config(raw_config)
        transition_name = f"transition_{index}"
        if transition_name not in arrays:
            raise ValueError(f"missing saved array {transition_name}")
        transition = numpy_to_tensor(
            arrays[transition_name], device=device, dtype=torch.float32
        )
        hidden_size = int(raw_config["hidden_size"])
        if transition.shape != (hidden_size, hidden_size):
            raise ValueError("saved Core transition table has the wrong shape")
        advancement = raw_config["advancement_period"]
        cores.append(
            _core_cls(
                num_states=hidden_size,
                transition_table=transition,
                advancement_period=advancement,
                dwell_sampler=_resolve_dwelltime_sampler(advancement, device, _probe=False),
                construction_config=dict(raw_config),
                device=device,
            )
        )
    return _agent_cls(cores, device)
