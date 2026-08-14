"""Generator provides optional whole-pipeline orchestration without absorbing component ownership.

The three reusable components remain public and independently persistable. Generator only constructs
or loads a compatible trio and applies Agent -> Language -> Vocabulary in order for callers who do
not need to manipulate those boundaries on every sample.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from ._validation import require_exact_keys
from .agent import Agent
from .compatibility import validate_compatibility
from .language.api import Language
from .persistence import ensure_directory, read_json, write_json
from .vocabulary import Vocabulary


# main


@dataclass(slots=True)
class GeneratedSample:
    """Carry the three aligned sample IRs produced by one Generator call."""

    latent: torch.Tensor
    grammar: torch.Tensor
    tokens: torch.Tensor


class Generator:
    """Orchestrate one compatible Agent, Language, and Vocabulary as a convenience facade."""

    def __init__(
        self,
        agent: Agent,
        language: Language,
        vocabulary: Vocabulary,
    ) -> None:
        validate_compatibility(agent, language, vocabulary)
        self.agent = agent
        self.language = language
        self.vocabulary = vocabulary

    @classmethod
    def from_config(
        cls,
        config: dict[str, object],
        device: str | torch.device,
    ) -> "Generator":
        """Construct a new compatible component trio from one top-level configuration."""
        return build_generator(config, torch.device(device), _generator_cls=cls)

    @classmethod
    def from_pretrained(
        cls,
        path: str | os.PathLike[str],
        device: str | torch.device,
        **runtime_overrides: int,
    ) -> "Generator":
        """Restore a saved trio, applying optional overrides only to Language runtime policy."""
        return load_generator_pretrained(
            path,
            torch.device(device),
            runtime_overrides,
            _generator_cls=cls,
        )

    def save_pretrained(self, path: str | os.PathLike[str]) -> None:
        """Persist the three components using their native pretrained formats."""
        save_generator_pretrained(self, path)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        length: int,
        initial_markov_states: torch.Tensor | None = None,
    ) -> GeneratedSample:
        """Generate aligned latent, grammar, and token samples through the component pipeline."""
        latent = self.agent.sample(
            batch_size=batch_size,
            length=length,
            initial_markov_states=initial_markov_states,
        )
        grammar = self.language.sample(latent)
        tokens = self.vocabulary.sample(latent, grammar)
        return GeneratedSample(latent=latent, grammar=grammar, tokens=tokens)

    __call__ = sample


# construction


def build_generator(
    config: dict[str, object],
    device: torch.device,
    _generator_cls: type[Generator] = Generator,
    _agent_cls: type[Agent] = Agent,
    _language_cls: type[Language] = Language,
    _vocabulary_cls: type[Vocabulary] = Vocabulary,
) -> Generator:
    """Construct the three public components and validate their explicit compatibility boundary."""
    require_exact_keys(config, {"agent", "language", "vocabulary"}, "Generator configuration")
    agent_config = config["agent"]
    language_config = config["language"]
    vocabulary_config = config["vocabulary"]
    if not isinstance(agent_config, dict) or not isinstance(language_config, dict) or not isinstance(vocabulary_config, dict):
        raise ValueError("Generator component configurations must be mappings")
    agent = _agent_cls.from_config(agent_config, device=device)
    language = _language_cls.from_config(language_config, device=device)
    vocabulary = _vocabulary_cls.from_config(language, vocabulary_config)
    return _generator_cls(agent, language, vocabulary)


def save_generator_pretrained(
    generator: Generator,
    path: str | os.PathLike[str],
) -> None:
    """Persist a Generator as a manifest plus native component subdirectories."""
    directory = ensure_directory(path)
    write_json(directory / "config.json", {"format": "latentseq-generator-v1"})
    generator.agent.save_pretrained(directory / "agent")
    generator.language.save_pretrained(directory / "language")
    generator.vocabulary.save_pretrained(directory / "vocabulary")


def load_generator_pretrained(
    path: str | os.PathLike[str],
    device: torch.device,
    runtime_overrides: dict[str, int],
    _generator_cls: type[Generator] = Generator,
    _agent_cls: type[Agent] = Agent,
    _language_cls: type[Language] = Language,
    _vocabulary_cls: type[Vocabulary] = Vocabulary,
) -> Generator:
    """Restore native component formats and revalidate the independently stored objects."""
    directory = Path(path)
    manifest = read_json(directory / "config.json")
    require_exact_keys(manifest, {"format"}, "Generator pretrained manifest")
    if manifest["format"] != "latentseq-generator-v1":
        raise ValueError("unsupported Generator pretrained format")
    agent = _agent_cls.from_pretrained(directory / "agent", device=device)
    language = _language_cls.from_pretrained(
        directory / "language", device=device, **runtime_overrides
    )
    vocabulary = _vocabulary_cls.from_pretrained(
        directory / "vocabulary", device=device
    )
    return _generator_cls(agent, language, vocabulary)
