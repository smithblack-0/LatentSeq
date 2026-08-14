"""Language IRs separate semantic language construction from backend execution.

`LSPCFG` retains the human-readable grammar and latent-conditioned semantic production tables.
`LanguageCore` retains the compiled Torch tables consumed by the pushdown runtime.  Both are passive
structures: neither owns execution state, trajectory state, or RNG state.
"""

from dataclasses import dataclass

import numpy as np
import torch


# main


@dataclass(slots=True)
class LSPCFG:
    """Represent one sampled latent-state-conditioned probabilistic grammar."""

    grammar: str
    probabilities: np.ndarray
    source_nodes: dict[int, str]
    sink_nodes: dict[int, tuple[int | str, ...]]
    source_to_sinks: dict[int, tuple[int, ...]]
    language_shape: dict[str, int]


@dataclass(slots=True)
class LanguageCore:
    """Represent one compiled static language executable for the Torch runtime."""

    transition_tables: torch.Tensor
    operand_table: torch.Tensor
    operation_table: torch.Tensor
    operand_length_table: torch.Tensor
    start_source_node: int
    done_source_node: int
    grammar_terminal_count: int
    semantic_source_to_runtime: dict[int, int]
    semantic_sink_to_runtime: dict[int, int]
    grammar_start_source_node: int
