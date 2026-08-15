"""Language integration tests run real compiled cores through all pushdown runtime components."""

import numpy as np
import pytest
import torch

from latentseq import Language
from latentseq.language.ir import LSPCFG


def _direct_language(device: torch.device) -> Language:
    probabilities = np.zeros((1, 2, 1, 2), dtype=np.float64)
    probabilities[0, 0, 0, 0] = 1.0
    probabilities[0, 1, 0, 1] = 1.0
    return Language.from_ls_pcfg(
        LSPCFG(
            grammar="direct",
            probabilities=probabilities,
            source_nodes={0: "A"},
            sink_nodes={0: (0,), 1: (1,)},
            source_to_sinks={0: (0, 1)},
            language_shape={"num_cores": 1, "max_num_states": 2},
        ),
        runtime_config={"stack_depth": 8, "chunk_size": 4, "max_attempts": 2},
        device=device,
    )


def test_direct_latent_to_terminal_lowering(device):
    language = _direct_language(device)
    latent = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.int64, device=device)
    assert language.sample(latent).tolist() == [[0, 1, 0, 1]]


def test_batch_parallel_direct_lowering(device):
    language = _direct_language(device)
    latent = torch.tensor([[[0, 0, 1]], [[1, 0, 1]]], dtype=torch.int64, device=device)
    assert language.sample(latent).tolist() == [[0, 0, 1], [1, 0, 1]]


def test_repeated_two_terminal_derivations_use_start_sentinel(device):
    probabilities = np.ones((1, 1, 1, 1), dtype=np.float64)
    language = Language.from_ls_pcfg(
        LSPCFG(
            grammar="two-terminal",
            probabilities=probabilities,
            source_nodes={0: "A"},
            sink_nodes={0: (0, 1)},
            source_to_sinks={0: (0,)},
            language_shape={"num_cores": 1, "max_num_states": 1},
        ),
        runtime_config={"stack_depth": 8, "chunk_size": 4, "max_attempts": 2},
        device=device,
    )
    latent = torch.zeros((1, 1, 6), dtype=torch.int64, device=device)
    assert language.sample(latent).tolist() == [[0, 1, 0, 1, 0, 1]]


def test_tiny_stack_rejects_overflow(device):
    probabilities = np.ones((1, 1, 1, 1), dtype=np.float64)
    language = Language.from_ls_pcfg(
        LSPCFG(
            grammar="overflow",
            probabilities=probabilities,
            source_nodes={0: "A"},
            sink_nodes={0: (0, 0, 0)},
            source_to_sinks={0: (0,)},
            language_shape={"num_cores": 1, "max_num_states": 1},
        ),
        runtime_config={"stack_depth": 2, "chunk_size": 4, "max_attempts": 2},
        device=device,
    )
    latent = torch.zeros((1, 1, 3), dtype=torch.int64, device=device)
    with pytest.raises(RuntimeError, match="stack"):
        language.sample(latent)


def test_completed_lane_stays_inert_while_other_lane_needs_more_expansions(device):
    probabilities = np.zeros((1, 2, 2, 3), dtype=np.float64)
    probabilities[0, 0, 0, 0] = 1.0  # A -> 0 directly for state 0.
    probabilities[0, 1, 0, 1] = 1.0  # A -> B for state 1.
    probabilities[:, :, 1, 2] = 1.0   # B -> 0 for every state.
    language = Language.from_ls_pcfg(
        LSPCFG(
            grammar="unequal-work",
            probabilities=probabilities,
            source_nodes={0: "A", 1: "B"},
            sink_nodes={0: (0,), 1: ("B",), 2: (0,)},
            source_to_sinks={0: (0, 1), 1: (2,)},
            language_shape={"num_cores": 1, "max_num_states": 2},
        ),
        runtime_config={"stack_depth": 8, "chunk_size": 8, "max_attempts": 2},
        device=device,
    )
    latent = torch.tensor(
        [
            [[0, 0, 0]],
            [[1, 1, 1]],
        ],
        dtype=torch.int64,
        device=device,
    )
    grammar = language.sample(latent)
    assert grammar.tolist() == [[0, 0, 0], [0, 0, 0]]


def test_two_core_latent_factor_conditioning_in_complete_runtime(device):
    probabilities = np.zeros((2, 2, 1, 2), dtype=np.float64)
    probabilities[0, 0, 0] = [0.8, 0.2]
    probabilities[0, 1, 0] = [0.2, 0.8]
    probabilities[1, 0, 0] = [1.0, 0.0]
    probabilities[1, 1, 0] = [0.0, 1.0]
    language = Language.from_ls_pcfg(
        LSPCFG(
            grammar="two-core",
            probabilities=probabilities,
            source_nodes={0: "A"},
            sink_nodes={0: (0,), 1: (1,)},
            source_to_sinks={0: (0, 1)},
            language_shape={"num_cores": 2, "max_num_states": 2},
        ),
        runtime_config={"stack_depth": 8, "chunk_size": 8, "max_attempts": 2},
        device=device,
    )
    latent = torch.tensor(
        [[[0, 1, 0, 1], [0, 1, 0, 1]]],
        dtype=torch.int64,
        device=device,
    )
    assert language.sample(latent).tolist() == [[0, 1, 0, 1]]
