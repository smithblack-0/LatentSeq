"""RNG boundary tests ensure LatentSeq consumes ambient streams without owning their lifecycle."""

import numpy as np
import torch

from latentseq import Agent, Language, Vocabulary


def test_external_numpy_seed_controls_language_construction(language_config, device):
    np.random.seed(44)
    first = Language.from_config(language_config, device=device)
    np.random.seed(44)
    second = Language.from_config(language_config, device=device)
    assert first.grammar == second.grammar
    assert np.array_equal(first.ls_pcfg.probabilities, second.ls_pcfg.probabilities)


def test_external_torch_seed_controls_agent_construction(agent_config, device):
    torch.manual_seed(9)
    first = Agent.from_config(agent_config, device=device)
    torch.manual_seed(9)
    second = Agent.from_config(agent_config, device=device)
    for left, right in zip(first.cores, second.cores, strict=True):
        assert torch.equal(left.transition_table, right.transition_table)


def test_pretrained_roundtrip_does_not_rewind_or_advance_rng(
    tmp_path, agent_config, language_config, vocabulary_config, device
):
    np.random.seed(2)
    torch.manual_seed(2)
    agent = Agent.from_config(agent_config, device=device)
    language = Language.from_config(language_config, device=device)
    vocabulary = Vocabulary.from_config(language, vocabulary_config)
    agent.save_pretrained(tmp_path / "agent")
    language.save_pretrained(tmp_path / "language")
    vocabulary.save_pretrained(tmp_path / "vocabulary")

    np.random.seed(700)
    torch.manual_seed(700)
    expected_numpy = np.random.random(3)
    expected_torch = torch.rand(3)

    np.random.seed(700)
    torch.manual_seed(700)
    Agent.from_pretrained(tmp_path / "agent", device=device)
    Language.from_pretrained(tmp_path / "language", device=device)
    Vocabulary.from_pretrained(tmp_path / "vocabulary", device=device)
    actual_numpy = np.random.random(3)
    actual_torch = torch.rand(3)

    assert np.array_equal(actual_numpy, expected_numpy)
    assert torch.equal(actual_torch, expected_torch)
