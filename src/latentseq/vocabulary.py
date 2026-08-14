"""Vocabulary maps grammar terminals into a large observable token space under latent conditioning.

A Vocabulary is sampled once against one Language identity. It owns fixed terminal bindings and one
raw lexical distribution per core/state/grammar terminal, then applies latent factor conditioning at
runtime to realize each Grammar Sample position as a concrete token. Persistence restores those
sampled tables exactly and never stores RNG or trajectory state.
"""

import math
import os
from pathlib import Path

import torch

from ._sampling_ops import sample_categorical

from ._validation import require_exact_keys, require_positive_int
from .language.api import Language
from .language.sampling import get_pseudotemperature
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

VOCABULARY_FIELDS = {"vocabulary_size", "elements_per_terminal", "pairwise_odds"}


def _validate_vocabulary_config(config: dict[str, object]) -> dict[str, int | float]:
    require_exact_keys(config, VOCABULARY_FIELDS, "Vocabulary configuration")
    vocabulary_size = require_positive_int(config["vocabulary_size"], "vocabulary_size")
    elements_per_terminal = require_positive_int(
        config["elements_per_terminal"], "elements_per_terminal"
    )
    pairwise_odds = config["pairwise_odds"]
    if (
        isinstance(pairwise_odds, bool)
        or not isinstance(pairwise_odds, (int, float))
        or pairwise_odds < 1
    ):
        raise ValueError("pairwise_odds must be >= 1")
    return {
        "vocabulary_size": vocabulary_size,
        "elements_per_terminal": elements_per_terminal,
        "pairwise_odds": float(pairwise_odds),
    }


def _sample_bindings(
    grammar_terminal_count: int,
    vocabulary_size: int,
    elements_per_terminal: int,
    device: torch.device,
) -> torch.Tensor:
    if elements_per_terminal > vocabulary_size:
        raise ValueError("elements_per_terminal cannot exceed vocabulary_size")
    if grammar_terminal_count * elements_per_terminal < vocabulary_size:
        raise ValueError(
            "grammar_terminal_count * elements_per_terminal must cover vocabulary_size"
        )
    bindings = torch.empty(
        (grammar_terminal_count, elements_per_terminal),
        dtype=torch.int64,
        device=device,
    )
    for terminal in range(grammar_terminal_count):
        bindings[terminal] = torch.randperm(
            vocabulary_size, device=device
        )[:elements_per_terminal]

    # Coverage repair is deterministic conditional on the sampled rows. Replacing an element whose
    # global count is >1 preserves coverage of the old ID while introducing one currently missing ID.
    counts = torch.bincount(bindings.flatten(), minlength=vocabulary_size)
    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    for missing_id in missing:
        repaired = False
        for terminal in range(grammar_terminal_count):
            row = bindings[terminal]
            if bool(torch.any(row == missing_id)):
                continue
            for slot in range(elements_per_terminal):
                old_id = int(row[slot])
                if int(counts[old_id]) > 1:
                    bindings[terminal, slot] = missing_id
                    counts[old_id] -= 1
                    counts[missing_id] += 1
                    repaired = True
                    break
            if repaired:
                break
        if not repaired:
            raise RuntimeError("Vocabulary coverage repair found no replaceable binding")
    return bindings


def _sample_probability_tables(
    num_cores: int,
    max_num_states: int,
    grammar_terminal_count: int,
    elements_per_terminal: int,
    pairwise_odds: float,
    device: torch.device,
) -> torch.Tensor:
    if elements_per_terminal == 1:
        return torch.ones(
            (num_cores, max_num_states, grammar_terminal_count, 1),
            dtype=torch.float32,
            device=device,
        )
    pseudo_temperature = get_pseudotemperature(
        {"pairwise_odds": pairwise_odds}, elements_per_terminal
    )
    core_std = pseudo_temperature / math.sqrt(num_cores)
    logits = torch.randn(
        (
            num_cores,
            max_num_states,
            grammar_terminal_count,
            elements_per_terminal,
        ),
        dtype=torch.float32,
        device=device,
    ) * core_std
    return torch.softmax(logits, dim=-1)


def _validate_runtime_inputs(
    vocabulary: "Vocabulary",
    latent_samples: torch.Tensor,
    grammar_samples: torch.Tensor,
) -> None:
    if latent_samples.ndim != 3 or latent_samples.dtype != torch.int64:
        raise ValueError("latent_samples must be int64 [batch,core,timestep]")
    if grammar_samples.ndim != 2 or grammar_samples.dtype != torch.int64:
        raise ValueError("grammar_samples must be int64 [batch,timestep]")
    if latent_samples.shape[0] != grammar_samples.shape[0] or latent_samples.shape[2] != grammar_samples.shape[1]:
        raise ValueError("latent_samples and grammar_samples must align on batch and timestep")
    if latent_samples.shape[1] != vocabulary.num_cores:
        raise ValueError("latent_samples core width is incompatible with Vocabulary")
    if latent_samples.device != vocabulary.device or grammar_samples.device != vocabulary.device:
        raise ValueError("Vocabulary inputs must be on the Vocabulary device")
    if bool(torch.any(latent_samples < 0)) or bool(
        torch.any(latent_samples >= vocabulary.max_num_states)
    ):
        raise ValueError("latent_samples contains an out-of-range state")
    if bool(torch.any(grammar_samples < 0)) or bool(
        torch.any(grammar_samples >= vocabulary.grammar_terminal_count)
    ):
        raise ValueError("grammar_samples contains an out-of-range terminal")


# main


class Vocabulary:
    """Expose one reusable lexical realization sampled against an exact Language identity.

    Vocabulary is independent runtime machinery once built: callers can persist/pool it separately,
    validate its Language fingerprint later, and apply it to aligned LS/GS samples without changing
    either source component.
    """

    def __init__(
        self,
        bindings: torch.Tensor,
        probabilities: torch.Tensor,
        construction_config: dict[str, int | float],
        language_fingerprint: str,
        num_cores: int,
        max_num_states: int,
        grammar_terminal_count: int,
        device: torch.device,
    ) -> None:
        self.bindings = bindings
        self.probabilities = probabilities
        self.construction_config = construction_config
        self.language_fingerprint = language_fingerprint
        self.num_cores = num_cores
        self.max_num_states = max_num_states
        self.grammar_terminal_count = grammar_terminal_count
        self.device = device
        self.sample_backend = (
            torch.compile(self._sample_unchecked, dynamic=False, fullgraph=True)
            if device.type == "cuda"
            else self._sample_unchecked
        )

    @classmethod
    def from_config(
        cls,
        language: Language,
        config: dict[str, object],
    ) -> "Vocabulary":
        """Sample a new Vocabulary compatible with one existing Language."""
        return build_vocabulary(language, config, _vocabulary_cls=cls)

    @classmethod
    def from_pretrained(
        cls,
        path: str | os.PathLike[str],
        device: str | torch.device,
    ) -> "Vocabulary":
        """Restore an exact saved Vocabulary on the selected execution device."""
        return load_vocabulary_pretrained(
            path, torch.device(device), _vocabulary_cls=cls
        )

    def save_pretrained(self, path: str | os.PathLike[str]) -> None:
        """Persist exact lexical tables and compatibility metadata, excluding RNG state."""
        save_vocabulary_pretrained(self, path)

    @torch.no_grad()
    def sample(
        self,
        latent_samples: torch.Tensor,
        grammar_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Realize aligned latent/grammar samples as int64 concrete tokens.

        Args:
            latent_samples: int64 `[batch,core,timestep]` latent states.
            grammar_samples: int64 `[batch,timestep]` grammar terminal IDs.

        Returns:
            int64 `[batch,timestep]` concrete vocabulary IDs on the Vocabulary device.
        """
        _validate_runtime_inputs(self, latent_samples, grammar_samples)
        return self.sample_backend(latent_samples, grammar_samples)

    @torch.no_grad()
    def _sample_unchecked(
        self,
        latent_samples: torch.Tensor,
        grammar_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the vectorized lexical lowering after public compatibility validation."""
        batch_size, _, timesteps = latent_samples.shape
        latent_by_core = latent_samples.permute(1, 0, 2)
        grammar_by_core = grammar_samples.unsqueeze(0).expand(
            self.num_cores, batch_size, timesteps
        )
        core_index = torch.arange(self.num_cores, device=self.device)[:, None, None]
        raw = self.probabilities[
            core_index, latent_by_core, grammar_by_core, :
        ]
        combined_logits = torch.log(raw).sum(dim=0)
        combined = torch.softmax(combined_logits, dim=-1)
        slots = sample_categorical(
            combined.reshape(batch_size * timesteps, -1)
        ).reshape(batch_size, timesteps)
        return self.bindings[grammar_samples, slots]

    __call__ = sample


# construction


def build_vocabulary(
    language: Language,
    config: dict[str, object],
    _vocabulary_cls: type[Vocabulary] = Vocabulary,
    _sample_bindings=_sample_bindings,
    _sample_probability_tables=_sample_probability_tables,
) -> Vocabulary:
    """Sample bindings and latent-conditioned lexical tables against one Language."""
    resolved = _validate_vocabulary_config(config)
    vocabulary_size = int(resolved["vocabulary_size"])
    elements_per_terminal = int(resolved["elements_per_terminal"])
    pairwise_odds = float(resolved["pairwise_odds"])
    bindings = _sample_bindings(
        language.grammar_terminal_count,
        vocabulary_size,
        elements_per_terminal,
        language.device,
    )
    probabilities = _sample_probability_tables(
        language.num_cores,
        language.max_num_states,
        language.grammar_terminal_count,
        elements_per_terminal,
        pairwise_odds,
        language.device,
    )
    return _vocabulary_cls(
        bindings=bindings,
        probabilities=probabilities,
        construction_config=resolved,
        language_fingerprint=language.fingerprint,
        num_cores=language.num_cores,
        max_num_states=language.max_num_states,
        grammar_terminal_count=language.grammar_terminal_count,
        device=language.device,
    )


def save_vocabulary_pretrained(
    vocabulary: Vocabulary,
    path: str | os.PathLike[str],
) -> None:
    """Persist one Vocabulary's exact sampled tables and compatibility identity."""
    directory = ensure_directory(path)
    write_json(
        directory / "config.json",
        {
            "format": "latentseq-vocabulary-v1",
            "construction_config": dict(vocabulary.construction_config),
            "language_fingerprint": vocabulary.language_fingerprint,
            "num_cores": vocabulary.num_cores,
            "max_num_states": vocabulary.max_num_states,
            "grammar_terminal_count": vocabulary.grammar_terminal_count,
        },
    )
    save_npz(
        directory / "state.npz",
        bindings=tensor_to_numpy(vocabulary.bindings),
        probabilities=tensor_to_numpy(vocabulary.probabilities),
    )


def load_vocabulary_pretrained(
    path: str | os.PathLike[str],
    device: torch.device,
    _vocabulary_cls: type[Vocabulary] = Vocabulary,
) -> Vocabulary:
    """Restore saved lexical tables directly without binding or probability resampling."""
    directory = Path(path)
    manifest = read_json(directory / "config.json")
    require_exact_keys(
        manifest,
        {
            "format",
            "construction_config",
            "language_fingerprint",
            "num_cores",
            "max_num_states",
            "grammar_terminal_count",
        },
        "Vocabulary pretrained manifest",
    )
    if manifest["format"] != "latentseq-vocabulary-v1":
        raise ValueError("unsupported Vocabulary pretrained format")
    raw_config = manifest["construction_config"]
    if not isinstance(raw_config, dict):
        raise ValueError("saved Vocabulary construction_config must be a mapping")
    config = _validate_vocabulary_config(raw_config)
    arrays = load_npz(directory / "state.npz")
    if set(arrays) != {"bindings", "probabilities"}:
        raise ValueError("saved Vocabulary state arrays are incomplete or unknown")
    bindings = numpy_to_tensor(arrays["bindings"], device, torch.int64)
    probabilities = numpy_to_tensor(arrays["probabilities"], device, torch.float32)
    num_cores = int(manifest["num_cores"])
    max_num_states = int(manifest["max_num_states"])
    grammar_terminal_count = int(manifest["grammar_terminal_count"])
    elements_per_terminal = int(config["elements_per_terminal"])
    if bindings.shape != (grammar_terminal_count, elements_per_terminal):
        raise ValueError("saved Vocabulary bindings have the wrong shape")
    expected_probability_shape = (
        num_cores,
        max_num_states,
        grammar_terminal_count,
        elements_per_terminal,
    )
    if probabilities.shape != expected_probability_shape:
        raise ValueError("saved Vocabulary probability table has the wrong shape")
    return _vocabulary_cls(
        bindings=bindings,
        probabilities=probabilities,
        construction_config=config,
        language_fingerprint=str(manifest["language_fingerprint"]),
        num_cores=num_cores,
        max_num_states=max_num_states,
        grammar_terminal_count=grammar_terminal_count,
        device=device,
    )
