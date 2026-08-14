"""Public API tests certify the low-ceremony interface normal researchers are expected to use."""

import numpy as np
import torch

from latentseq import Agent, Generator, Language, Vocabulary, validate_compatibility


def _seed_external(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_end_to_end_public_usage(agent_config, language_config, vocabulary_config, device):
    _seed_external(1234)
    agent = Agent.from_config(agent_config, device=device)
    language = Language.from_config(language_config, device=device)
    vocabulary = Vocabulary.from_config(language, vocabulary_config)

    validate_compatibility(agent, language, vocabulary)

    latent = agent.sample(batch_size=3, length=24)
    grammar = language.sample(latent)
    tokens = vocabulary.sample(latent, grammar)

    assert latent.shape == (3, 2, 24)
    assert grammar.shape == (3, 24)
    assert tokens.shape == (3, 24)
    assert latent.dtype == grammar.dtype == tokens.dtype == torch.int64


def test_generator_is_only_orchestration(agent_config, language_config, vocabulary_config, device):
    _seed_external(123)
    generator = Generator.from_config(
        {
            "agent": agent_config,
            "language": language_config,
            "vocabulary": vocabulary_config,
        },
        device=device,
    )
    sample = generator.sample(batch_size=2, length=12)
    assert sample.latent.shape == (2, 2, 12)
    assert sample.grammar.shape == (2, 12)
    assert sample.tokens.shape == (2, 12)
    assert generator.agent is not None
    assert generator.language is not None
    assert generator.vocabulary is not None


def test_top_level_surface_exposes_user_objects_not_runtime_helpers():
    import latentseq

    for name in ["Agent", "Language", "Vocabulary", "Generator", "validate_compatibility"]:
        assert hasattr(latentseq, name)
    for name in [
        "Stack",
        "Reader",
        "Recorder",
        "CounterDecoder",
        "Transitions",
        "InstructionDecoder",
        "set_seed",
    ]:
        assert not hasattr(latentseq, name)
