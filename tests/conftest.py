"""Shared test configurations keep LatentSeq behavior tests small enough to audit by inspection."""

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def agent_config() -> dict:
    return {
        "cores": [
            {"advancement_period": 0},
            {"advancement_period": 2},
        ],
        "defaults": {
            "hidden_size": 4,
            "num_connections": 2,
            "concentration": 1.0,
        },
    }


@pytest.fixture
def language_config() -> dict:
    return {
        "grammar": {
            "terminal_pair_rules": 4,
            "parenthesis_rules": 2,
            "iteration_rules": 2,
            "branch_rules": 1,
            "max_terminals": 6,
            "max_nonterminals": 6,
            "language_shape": {"num_cores": 2, "max_num_states": 4},
            "sampling_defaults": {
                "pairwise_odds": 3.0,
                "min_ppl": 1.1,
                "max_ppl": 5.0,
            },
        },
        "runtime": {
            "stack_depth": 32,
            "chunk_size": 16,
            "max_attempts": 4,
        },
    }


@pytest.fixture
def vocabulary_config() -> dict:
    return {
        "vocabulary_size": 5,
        "elements_per_terminal": 5,
        "pairwise_odds": 2.0,
    }
