"""Vocabulary tests certify binding coverage, latent factor conditioning, alignment, and persistence identity."""

import numpy as np
import pytest
import torch

from latentseq import Language, Vocabulary


def test_binding_width_coverage_and_caller_seed_reproducibility(language_config, vocabulary_config, device):
    np.random.seed(21)
    torch.manual_seed(21)
    language_a = Language.from_config(language_config, device=device)
    vocabulary_a = Vocabulary.from_config(language_a, vocabulary_config)
    np.random.seed(21)
    torch.manual_seed(21)
    language_b = Language.from_config(language_config, device=device)
    vocabulary_b = Vocabulary.from_config(language_b, vocabulary_config)

    k = vocabulary_config["elements_per_terminal"]
    assert vocabulary_a.bindings.shape[1] == k
    assert all(len(set(row.tolist())) == k for row in vocabulary_a.bindings)
    assert set(vocabulary_a.bindings.flatten().tolist()) == set(range(vocabulary_config["vocabulary_size"]))
    assert torch.equal(vocabulary_a.bindings, vocabulary_b.bindings)
    assert torch.equal(vocabulary_a.probabilities, vocabulary_b.probabilities)


def test_probability_rows_are_normalized(language_config, vocabulary_config, device):
    language = Language.from_config(language_config, device=device)
    vocabulary = Vocabulary.from_config(language, vocabulary_config)
    assert torch.allclose(
        vocabulary.probabilities.sum(dim=-1),
        torch.ones_like(vocabulary.probabilities.sum(dim=-1)),
    )


def test_vocabulary_alignment(language_config, vocabulary_config, device):
    language = Language.from_config(language_config, device=device)
    vocabulary = Vocabulary.from_config(language, vocabulary_config)
    batch, length = 2, 5
    latent = torch.zeros((batch, 2, length), dtype=torch.int64, device=device)
    grammar = torch.arange(length, device=device).remainder(language.grammar_terminal_count).repeat(batch, 1)
    tokens = vocabulary.sample(latent, grammar)
    assert tokens.shape == grammar.shape
    for batch_index in range(batch):
        for timestep in range(length):
            assert tokens[batch_index, timestep].item() in vocabulary.bindings[grammar[batch_index, timestep]].tolist()


def test_hand_probability_fixture_intersects_selected_core_rows(device):
    # Use a tiny real Language only to establish compatibility metadata, then inject deterministic
    # lexical tables to isolate runtime Vocabulary semantics.
    probabilities = np.ones((2, 1, 1, 1), dtype=np.float64)
    from latentseq.language.ir import LSPCFG

    language = Language.from_ls_pcfg(
        LSPCFG(
            grammar="tiny",
            probabilities=probabilities,
            source_nodes={0: "A"},
            sink_nodes={0: (0,)},
            source_to_sinks={0: (0,)},
            language_shape={"num_cores": 2, "max_num_states": 1},
        ),
        {"stack_depth": 4, "chunk_size": 2, "max_attempts": 1},
        device,
    )
    vocab = Vocabulary(
        bindings=torch.tensor([[10, 11, 12]], dtype=torch.int64, device=device),
        probabilities=torch.tensor(
            [
                [[[0.5, 0.5, 0.0]]],
                [[[0.0, 0.5, 0.5]]],
            ],
            dtype=torch.float32,
            device=device,
        ),
        construction_config={"vocabulary_size": 13, "elements_per_terminal": 3, "pairwise_odds": 1.0},
        language_fingerprint=language.fingerprint,
        num_cores=2,
        max_num_states=1,
        grammar_terminal_count=1,
        device=device,
    )
    latent = torch.zeros((2, 2, 4), dtype=torch.int64, device=device)
    grammar = torch.zeros((2, 4), dtype=torch.int64, device=device)
    # Only slot 1 survives the product, so every output must be binding 11.
    assert torch.equal(vocab.sample(latent, grammar), torch.full((2, 4), 11, dtype=torch.int64, device=device))


def test_vocabulary_rejects_infeasible_binding_configuration(language_config, device):
    language = Language.from_config(language_config, device=device)
    with pytest.raises(ValueError):
        Vocabulary.from_config(
            language,
            {"vocabulary_size": 3, "elements_per_terminal": 4, "pairwise_odds": 2.0},
        )
    with pytest.raises(ValueError):
        Vocabulary.from_config(
            language,
            {
                "vocabulary_size": language.grammar_terminal_count * 3 + 1,
                "elements_per_terminal": 3,
                "pairwise_odds": 2.0,
            },
        )
