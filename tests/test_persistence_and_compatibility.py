"""Persistence and compatibility tests certify independent pools can be restored and recombined safely."""

import numpy as np
import pytest
import torch

from latentseq import (
    Agent,
    CompatibilityError,
    Generator,
    Language,
    Vocabulary,
    validate_compatibility,
)


def test_pretrained_round_trip_and_runtime_overrides(tmp_path, agent_config, language_config, vocabulary_config, device):
    np.random.seed(8)
    torch.manual_seed(8)
    agent = Agent.from_config(agent_config, device=device)
    language = Language.from_config(language_config, device=device)
    vocabulary = Vocabulary.from_config(language, vocabulary_config)

    agent.save_pretrained(tmp_path / "agent")
    language.save_pretrained(tmp_path / "language")
    vocabulary.save_pretrained(tmp_path / "vocabulary")

    restored_agent = Agent.from_pretrained(tmp_path / "agent", device=device)
    restored_language = Language.from_pretrained(
        tmp_path / "language", device=device, stack_depth=64, chunk_size=8
    )
    restored_vocabulary = Vocabulary.from_pretrained(tmp_path / "vocabulary", device=device)

    assert restored_agent.state_signature == agent.state_signature
    for left, right in zip(restored_agent.cores, agent.cores, strict=True):
        assert torch.equal(left.transition_table, right.transition_table)
    assert restored_language.fingerprint == language.fingerprint
    assert np.array_equal(restored_language.ls_pcfg.probabilities, language.ls_pcfg.probabilities)
    assert torch.equal(restored_language.core.transition_tables, language.core.transition_tables)
    assert restored_language.runtime_config == {
        "stack_depth": 64,
        "chunk_size": 8,
        "max_attempts": language.runtime_config["max_attempts"],
    }
    assert torch.equal(restored_vocabulary.bindings, vocabulary.bindings)
    assert torch.equal(restored_vocabulary.probabilities, vocabulary.probabilities)
    validate_compatibility(restored_agent, restored_language, restored_vocabulary)


def test_language_pretrained_rejects_construction_overrides(tmp_path, language_config, device):
    language = Language.from_config(language_config, device=device)
    language.save_pretrained(tmp_path)
    with pytest.raises(TypeError, match="runtime overrides only"):
        Language.from_pretrained(tmp_path, device=device, pairwise_odds=10)


def test_language_vocabulary_identity_mismatch_is_reported(language_config, vocabulary_config, device):
    np.random.seed(1)
    language_a = Language.from_config(language_config, device=device)
    vocabulary = Vocabulary.from_config(language_a, vocabulary_config)
    np.random.seed(2)
    language_b = Language.from_config(language_config, device=device)
    with pytest.raises(CompatibilityError, match="fingerprint"):
        validate_compatibility(language_b, vocabulary)


def test_agent_language_core_count_mismatch_is_reported(agent_config, language_config, device):
    bad_agent = Agent.from_config(
        {"cores": [{"advancement_period": 0}], "defaults": agent_config["defaults"]},
        device=device,
    )
    language = Language.from_config(language_config, device=device)
    with pytest.raises(CompatibilityError, match="core count"):
        validate_compatibility(bad_agent, language)


def test_agent_state_width_mismatch_is_reported(agent_config, language_config, device):
    bad_agent = Agent.from_config(
        {
            "cores": agent_config["cores"],
            "defaults": {**agent_config["defaults"], "hidden_size": 8},
        },
        device=device,
    )
    language = Language.from_config(language_config, device=device)
    with pytest.raises(CompatibilityError, match="state width"):
        validate_compatibility(bad_agent, language)


def test_compatibility_accepts_useful_subsets(agent_config, language_config, vocabulary_config, device):
    agent = Agent.from_config(agent_config, device=device)
    language = Language.from_config(language_config, device=device)
    vocabulary = Vocabulary.from_config(language, vocabulary_config)
    validate_compatibility(agent, language)
    validate_compatibility(language, vocabulary)
    validate_compatibility(agent, vocabulary)


def test_generator_pretrained_composes_native_component_formats(tmp_path, agent_config, language_config, vocabulary_config, device):
    generator = Generator.from_config(
        {"agent": agent_config, "language": language_config, "vocabulary": vocabulary_config},
        device=device,
    )
    generator.save_pretrained(tmp_path)
    restored = Generator.from_pretrained(tmp_path, device=device, chunk_size=4)
    assert restored.language.runtime_config["chunk_size"] == 4
    validate_compatibility(restored.agent, restored.language, restored.vocabulary)
