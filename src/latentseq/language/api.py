"""Public Language orchestration turns a language configuration into a reusable executable object.

Normal callers use `Language.from_config`, `sample`, `save_pretrained`, and `from_pretrained` without
manually carrying CFG, LS-PCFG, or LanguageCore stages. The semantic IRs are retained so advanced
analysis remains available, while runtime generation delegates to the decomposed pushdown helpers.
RNG state is neither stored nor restored; stochastic calls consume ambient library RNG streams.
"""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from latentseq._validation import require_exact_keys, require_positive_int
from latentseq.persistence import (
    ensure_directory,
    load_npz,
    numpy_to_tensor,
    read_json,
    save_npz,
    tensor_to_numpy,
    write_json,
)

from .analysis import LanguageAnalysis, analyze_language
from .cfg import sample_cfg
from .compilation import compile_language_core
from .ir import LSPCFG, LanguageCore
from .runtime import (
    InstructionDecoder,
    RuntimeAttempt,
    Transitions,
    build_instruction_decoder,
    build_runtime_attempt,
    build_transitions,
    select_decode_chunk_backend,
)
from .sampling import sample_ls_pcfg


# helpers

LANGUAGE_CONFIG_FIELDS = {"grammar", "runtime"}
RUNTIME_FIELDS = {"stack_depth", "chunk_size", "max_attempts"}


def _validate_runtime_config(config: dict[str, object]) -> dict[str, int]:
    require_exact_keys(config, RUNTIME_FIELDS, "Language runtime configuration")
    return {
        key: require_positive_int(config[key], key)
        for key in ("stack_depth", "chunk_size", "max_attempts")
    }


def _canonical_semantic_payload(ls_pcfg: LSPCFG) -> bytes:
    metadata = {
        "grammar": ls_pcfg.grammar,
        "source_nodes": [[index, ls_pcfg.source_nodes[index]] for index in sorted(ls_pcfg.source_nodes)],
        "sink_nodes": [
            [index, list(ls_pcfg.sink_nodes[index])] for index in sorted(ls_pcfg.sink_nodes)
        ],
        "source_to_sinks": [
            [index, list(ls_pcfg.source_to_sinks[index])]
            for index in sorted(ls_pcfg.source_to_sinks)
        ],
        "language_shape": ls_pcfg.language_shape,
        "probability_shape": list(ls_pcfg.probabilities.shape),
        "probability_dtype": str(ls_pcfg.probabilities.dtype),
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return encoded + b"\0" + np.ascontiguousarray(ls_pcfg.probabilities).tobytes()


def language_fingerprint(ls_pcfg: LSPCFG) -> str:
    """Return a deterministic identity hash for one exact semantic sampled language."""
    return hashlib.sha256(_canonical_semantic_payload(ls_pcfg)).hexdigest()


def _encode_ls_pcfg(ls_pcfg: LSPCFG) -> dict[str, object]:
    return {
        "source_nodes": [[index, value] for index, value in sorted(ls_pcfg.source_nodes.items())],
        "sink_nodes": [[index, list(value)] for index, value in sorted(ls_pcfg.sink_nodes.items())],
        "source_to_sinks": [
            [index, list(value)] for index, value in sorted(ls_pcfg.source_to_sinks.items())
        ],
        "language_shape": dict(ls_pcfg.language_shape),
    }


def _decode_ls_pcfg(
    grammar: str,
    metadata: dict[str, object],
    probabilities: np.ndarray,
) -> LSPCFG:
    require_exact_keys(
        metadata,
        {"source_nodes", "sink_nodes", "source_to_sinks", "language_shape"},
        "saved LS-PCFG metadata",
    )
    source_nodes = {
        int(index): str(value) for index, value in metadata["source_nodes"]  # type: ignore[misc]
    }
    sink_nodes = {
        int(index): tuple(value) for index, value in metadata["sink_nodes"]  # type: ignore[misc]
    }
    source_to_sinks = {
        int(index): tuple(int(sink) for sink in value)
        for index, value in metadata["source_to_sinks"]  # type: ignore[misc]
    }
    language_shape = metadata["language_shape"]
    if not isinstance(language_shape, dict):
        raise ValueError("saved Language shape must be a mapping")
    return LSPCFG(
        grammar=grammar,
        probabilities=probabilities.astype(np.float64, copy=False),
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
        source_to_sinks=source_to_sinks,
        language_shape={
            "num_cores": int(language_shape["num_cores"]),
            "max_num_states": int(language_shape["max_num_states"]),
        },
    )


def _encode_core(core: LanguageCore) -> dict[str, object]:
    return {
        "start_source_node": core.start_source_node,
        "done_source_node": core.done_source_node,
        "grammar_terminal_count": core.grammar_terminal_count,
        "semantic_source_to_runtime": [
            [key, value] for key, value in sorted(core.semantic_source_to_runtime.items())
        ],
        "semantic_sink_to_runtime": [
            [key, value] for key, value in sorted(core.semantic_sink_to_runtime.items())
        ],
        "grammar_start_source_node": core.grammar_start_source_node,
    }


def _decode_core(
    metadata: dict[str, object],
    arrays: dict[str, np.ndarray],
    device: torch.device,
) -> LanguageCore:
    require_exact_keys(
        metadata,
        {
            "start_source_node",
            "done_source_node",
            "grammar_terminal_count",
            "semantic_source_to_runtime",
            "semantic_sink_to_runtime",
            "grammar_start_source_node",
        },
        "saved LanguageCore metadata",
    )
    required_arrays = {
        "transition_tables",
        "operand_table",
        "operation_table",
        "operand_length_table",
    }
    missing = required_arrays - set(arrays)
    if missing:
        raise ValueError(f"saved Language is missing arrays: {sorted(missing)}")
    return LanguageCore(
        transition_tables=numpy_to_tensor(
            arrays["transition_tables"], device, torch.float32
        ),
        operand_table=numpy_to_tensor(arrays["operand_table"], device, torch.int64),
        operation_table=numpy_to_tensor(
            arrays["operation_table"], device, torch.int64
        ),
        operand_length_table=numpy_to_tensor(
            arrays["operand_length_table"], device, torch.int64
        ),
        start_source_node=int(metadata["start_source_node"]),
        done_source_node=int(metadata["done_source_node"]),
        grammar_terminal_count=int(metadata["grammar_terminal_count"]),
        semantic_source_to_runtime={
            int(key): int(value)
            for key, value in metadata["semantic_source_to_runtime"]  # type: ignore[misc]
        },
        semantic_sink_to_runtime={
            int(key): int(value)
            for key, value in metadata["semantic_sink_to_runtime"]  # type: ignore[misc]
        },
        grammar_start_source_node=int(metadata["grammar_start_source_node"]),
    )


def _validate_latent_for_language(
    latent_samples: torch.Tensor,
    language: "Language",
) -> None:
    if latent_samples.ndim != 3 or latent_samples.dtype != torch.int64:
        raise ValueError("latent_samples must be int64 [batch,core,timestep]")
    if latent_samples.device != language.device:
        raise ValueError("latent_samples must be on the Language device")
    expected_cores = language.core.transition_tables.shape[0]
    if latent_samples.shape[1] != expected_cores:
        raise ValueError(
            f"latent_samples has {latent_samples.shape[1]} cores; Language expects {expected_cores}"
        )
    max_states = language.core.transition_tables.shape[1]
    if bool(torch.any(latent_samples < 0)) or bool(torch.any(latent_samples >= max_states)):
        raise ValueError("latent_samples contains a state outside the Language state width")


# main


class Language:
    """Expose one sampled language as a reusable latent-to-grammar sequence transformer.

    Language keeps the semantic LS-PCFG for analysis/provenance and the compiled LanguageCore for
    execution. Construction-time stochastic identity is fixed once created; runtime stack/chunk/retry
    policy can be changed when restoring a pretrained Language.
    """

    def __init__(
        self,
        ls_pcfg: LSPCFG,
        core: LanguageCore,
        runtime_config: dict[str, int],
        fingerprint: str,
        transitions: Transitions,
        instruction_decoder: InstructionDecoder,
        attempt_factory: Callable[[torch.Tensor, LanguageCore, int], RuntimeAttempt],
        decode_chunk_function: Callable[[RuntimeAttempt, Transitions, InstructionDecoder, int], None],
        device: torch.device,
    ) -> None:
        self.ls_pcfg = ls_pcfg
        self.core = core
        self.runtime_config = runtime_config
        self.fingerprint = fingerprint
        self.transitions = transitions
        self.instruction_decoder = instruction_decoder
        self.attempt_factory = attempt_factory
        self.decode_chunk_function = decode_chunk_function
        self.device = device

    @property
    def grammar(self) -> str:
        """Return the human-readable CFG retained by this sampled Language."""
        return self.ls_pcfg.grammar

    @property
    def grammar_terminal_count(self) -> int:
        """Return the dense semantic terminal width produced by this Language."""
        return self.core.grammar_terminal_count

    @property
    def num_cores(self) -> int:
        """Return the ordered latent-core count consumed by this Language."""
        return int(self.core.transition_tables.shape[0])

    @property
    def max_num_states(self) -> int:
        """Return the Language-wide latent-state table width."""
        return int(self.core.transition_tables.shape[1])

    @classmethod
    def from_config(
        cls,
        config: dict[str, object],
        device: str | torch.device,
    ) -> "Language":
        """Construct and compile a newly sampled Language from ordinary user configuration."""
        return build_language_from_config(config, torch.device(device), _language_cls=cls)

    @classmethod
    def from_ls_pcfg(
        cls,
        ls_pcfg: LSPCFG,
        runtime_config: dict[str, object],
        device: str | torch.device,
    ) -> "Language":
        """Construct an executable Language from an already sampled semantic LS-PCFG."""
        return build_language_from_ls_pcfg(
            ls_pcfg, runtime_config, torch.device(device), _language_cls=cls
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | os.PathLike[str],
        device: str | torch.device,
        **runtime_overrides: int,
    ) -> "Language":
        """Restore an exact sampled Language with optional runtime-only policy overrides."""
        return load_language_pretrained(
            path,
            torch.device(device),
            runtime_overrides,
            _language_cls=cls,
        )

    def save_pretrained(self, path: str | os.PathLike[str]) -> None:
        """Persist exact semantic/compiled language state and runtime policy, excluding RNG state."""
        save_language_pretrained(self, path)

    def analyze(self, analysis_configuration: dict[str, object]) -> LanguageAnalysis:
        """Analyze this Language's retained LS-PCFG without entering the runtime path."""
        return analyze_language(self.ls_pcfg, analysis_configuration)

    @torch.no_grad()
    def sample(self, latent_samples: torch.Tensor) -> torch.Tensor:
        """Lower one aligned Latent Sample into a Grammar Sample `[batch,timestep]`.

        Args:
            latent_samples: int64 `[batch,core,timestep]` on the Language device.

        Returns:
            int64 Grammar Sample on the same device. Stack overflow discards the entire attempt and
            retries with fresh attempt state until `max_attempts` is exhausted.
        """
        _validate_latent_for_language(latent_samples, self)
        stack_depth = self.runtime_config["stack_depth"]
        chunk_size = self.runtime_config["chunk_size"]
        max_attempts = self.runtime_config["max_attempts"]

        for _ in range(max_attempts):
            attempt = self.attempt_factory(latent_samples, self.core, stack_depth)
            while True:
                self.decode_chunk_function(
                    attempt,
                    self.transitions,
                    self.instruction_decoder,
                    chunk_size,
                )
                # Host-side decisions happen only at the deliberate chunk boundary. Overflow takes
                # precedence over completion because any overflow invalidates the entire attempt.
                if bool(torch.any(attempt.stack.overflow)):
                    break
                if bool(torch.all(attempt.counter.is_complete())):
                    return attempt.recorder.retrieve()
        raise RuntimeError("Language generation repeatedly exceeded stack capacity")

    __call__ = sample


# construction


def build_language_from_ls_pcfg(
    ls_pcfg: LSPCFG,
    runtime_config: dict[str, object],
    device: torch.device,
    _language_cls: type[Language] = Language,
    _compile_language_core=compile_language_core,
    _build_transitions=build_transitions,
    _build_instruction_decoder=build_instruction_decoder,
    _attempt_factory=build_runtime_attempt,
    _select_decode_chunk_backend=select_decode_chunk_backend,
) -> Language:
    """Compile one semantic language and assemble the public runtime orchestration object."""
    resolved_runtime = _validate_runtime_config(runtime_config)
    core = _compile_language_core(ls_pcfg, device)
    return _language_cls(
        ls_pcfg=ls_pcfg,
        core=core,
        runtime_config=resolved_runtime,
        fingerprint=language_fingerprint(ls_pcfg),
        transitions=_build_transitions(core),
        instruction_decoder=_build_instruction_decoder(core),
        attempt_factory=_attempt_factory,
        decode_chunk_function=_select_decode_chunk_backend(device),
        device=device,
    )


def build_language_from_config(
    config: dict[str, object],
    device: torch.device,
    _language_cls: type[Language] = Language,
    _sample_cfg=sample_cfg,
    _sample_ls_pcfg=sample_ls_pcfg,
    _build_language_from_ls_pcfg=build_language_from_ls_pcfg,
) -> Language:
    """Orchestrate CFG sampling, Language Sampling, compilation, and public Language creation."""
    require_exact_keys(config, LANGUAGE_CONFIG_FIELDS, "Language configuration")
    grammar_config = config["grammar"]
    runtime_config = config["runtime"]
    if not isinstance(grammar_config, dict) or not isinstance(runtime_config, dict):
        raise ValueError("Language grammar and runtime entries must be mappings")
    grammar = _sample_cfg(grammar_config)
    ls_pcfg = _sample_ls_pcfg(grammar)
    return _build_language_from_ls_pcfg(
        ls_pcfg,
        runtime_config,
        device,
        _language_cls=_language_cls,
    )


def save_language_pretrained(
    language: Language,
    path: str | os.PathLike[str],
) -> None:
    """Persist semantic and compiled Language state atomically at the file level."""
    directory = ensure_directory(path)
    (directory / "grammar.txt").write_text(language.ls_pcfg.grammar, encoding="utf-8")
    write_json(
        directory / "config.json",
        {
            "format": "latentseq-language-v1",
            "fingerprint": language.fingerprint,
            "runtime_config": dict(language.runtime_config),
            "ls_pcfg": _encode_ls_pcfg(language.ls_pcfg),
            "core": _encode_core(language.core),
        },
    )
    save_npz(
        directory / "state.npz",
        ls_probabilities=language.ls_pcfg.probabilities,
        transition_tables=tensor_to_numpy(language.core.transition_tables),
        operand_table=tensor_to_numpy(language.core.operand_table),
        operation_table=tensor_to_numpy(language.core.operation_table),
        operand_length_table=tensor_to_numpy(language.core.operand_length_table),
    )


def load_language_pretrained(
    path: str | os.PathLike[str],
    device: torch.device,
    runtime_overrides: dict[str, int],
    _language_cls: type[Language] = Language,
    _build_transitions=build_transitions,
    _build_instruction_decoder=build_instruction_decoder,
    _attempt_factory=build_runtime_attempt,
    _select_decode_chunk_backend=select_decode_chunk_backend,
) -> Language:
    """Restore saved semantic/compiled state directly and apply only runtime-policy overrides."""
    unknown_overrides = set(runtime_overrides) - RUNTIME_FIELDS
    if unknown_overrides:
        raise TypeError(
            f"Language.from_pretrained accepts runtime overrides only; unknown={sorted(unknown_overrides)}"
        )
    directory = Path(path)
    manifest = read_json(directory / "config.json")
    require_exact_keys(
        manifest,
        {"format", "fingerprint", "runtime_config", "ls_pcfg", "core"},
        "Language pretrained manifest",
    )
    if manifest["format"] != "latentseq-language-v1":
        raise ValueError("unsupported Language pretrained format")
    saved_runtime = manifest["runtime_config"]
    if not isinstance(saved_runtime, dict):
        raise ValueError("saved Language runtime_config must be a mapping")
    runtime_config = _validate_runtime_config({**saved_runtime, **runtime_overrides})
    arrays = load_npz(directory / "state.npz")
    if "ls_probabilities" not in arrays:
        raise ValueError("saved Language is missing LS-PCFG probabilities")
    grammar = (directory / "grammar.txt").read_text(encoding="utf-8")
    ls_metadata = manifest["ls_pcfg"]
    core_metadata = manifest["core"]
    if not isinstance(ls_metadata, dict) or not isinstance(core_metadata, dict):
        raise ValueError("saved Language metadata is malformed")
    ls_pcfg = _decode_ls_pcfg(grammar, ls_metadata, arrays["ls_probabilities"])
    fingerprint = language_fingerprint(ls_pcfg)
    if fingerprint != manifest["fingerprint"]:
        raise ValueError("saved Language semantic fingerprint does not match its state")
    core = _decode_core(core_metadata, arrays, device)
    return _language_cls(
        ls_pcfg=ls_pcfg,
        core=core,
        runtime_config=runtime_config,
        fingerprint=fingerprint,
        transitions=_build_transitions(core),
        instruction_decoder=_build_instruction_decoder(core),
        attempt_factory=_attempt_factory,
        decode_chunk_function=_select_decode_chunk_backend(device),
        device=device,
    )
