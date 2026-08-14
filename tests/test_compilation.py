"""Compilation tests protect shared numbering, probability transfer, terminal identity, and LIFO instructions."""

import numpy as np
import pytest
import torch

from latentseq.language.compilation import NOOP, OUTPUT, PUSH, compile_language_core
from latentseq.language.ir import LSPCFG


def _fixture() -> LSPCFG:
    probabilities = np.zeros((2, 2, 2, 3), dtype=np.float64)
    probabilities[:, :, 0, 0] = 0.7
    probabilities[:, :, 0, 1] = 0.3
    probabilities[:, :, 1, 2] = 1.0
    return LSPCFG(
        grammar="fixture",
        probabilities=probabilities,
        source_nodes={0: "A", 1: "B"},
        sink_nodes={0: (0, "B", 1), 1: (1,), 2: (0,)},
        source_to_sinks={0: (0, 1), 1: (2,)},
        language_shape={"num_cores": 2, "max_num_states": 2},
    )


def test_compilation_preserves_shape_probabilities_and_instruction_order(device):
    fixture = _fixture()
    core = compile_language_core(fixture, device)
    assert core.transition_tables.shape[:2] == (2, 2)
    assert core.transition_tables.dtype == torch.float32
    source = core.semantic_source_to_runtime[0]
    sink = core.semantic_sink_to_runtime[0]
    assert torch.allclose(core.transition_tables[:, :, source, sink], torch.full((2, 2), 0.7))
    assert core.operation_table[sink].item() == PUSH
    assert core.operand_length_table[sink].item() == 3
    operands = core.operand_table[sink, :3].tolist()
    # Original RHS 0 B 1 executes in that order; stack stores reversed executor/source indexes.
    assert operands[1] == core.semantic_source_to_runtime[1]
    assert operands[0] != 0 and operands[2] != 1


def test_singleton_terminal_outputs_identity_without_executor_requirement(device):
    probabilities = np.ones((1, 1, 1, 1), dtype=np.float64)
    fixture = LSPCFG(
        grammar="singleton",
        probabilities=probabilities,
        source_nodes={0: "A"},
        sink_nodes={0: (0,)},
        source_to_sinks={0: (0,)},
        language_shape={"num_cores": 1, "max_num_states": 1},
    )
    core = compile_language_core(fixture, device)
    sink = core.semantic_sink_to_runtime[0]
    assert core.operation_table[sink].item() == OUTPUT
    assert core.operand_table[sink, 0].item() == 0
    assert core.operand_length_table[sink].item() == 1


def test_start_and_done_control_rows_are_deterministic(device):
    core = compile_language_core(_fixture(), device)
    start_rows = core.transition_tables[:, :, core.start_source_node]
    done_rows = core.transition_tables[:, :, core.done_source_node]
    assert torch.allclose(start_rows.sum(-1), torch.ones_like(start_rows.sum(-1)))
    assert torch.allclose(done_rows.sum(-1), torch.ones_like(done_rows.sum(-1)))
    start_sink = torch.argmax(start_rows[0, 0]).item()
    done_sink = torch.argmax(done_rows[0, 0]).item()
    assert core.operation_table[start_sink].item() == PUSH
    assert core.operand_table[start_sink, :2].tolist() == [core.start_source_node, core.grammar_start_source_node]
    assert core.operation_table[done_sink].item() == NOOP


def test_compilation_is_deterministic(device):
    first = compile_language_core(_fixture(), device)
    second = compile_language_core(_fixture(), device)
    assert torch.equal(first.transition_tables, second.transition_tables)
    assert torch.equal(first.operand_table, second.operand_table)
    assert first.semantic_source_to_runtime == second.semantic_source_to_runtime
    assert first.semantic_sink_to_runtime == second.semantic_sink_to_runtime


def test_compilation_rejects_gapped_terminal_ids(device):
    fixture = _fixture()
    fixture.sink_nodes = {0: (0, "B", 2), 1: (2,), 2: (0,)}
    with pytest.raises(ValueError, match="dense"):
        compile_language_core(fixture, device)


def test_compound_rhs_occurrences_reuse_one_terminal_executor(device):
    probabilities = np.zeros((1, 1, 2, 2), dtype=np.float64)
    probabilities[0, 0, 0, 0] = 1.0
    probabilities[0, 0, 1, 1] = 1.0
    fixture = LSPCFG(
        grammar="shared executor",
        probabilities=probabilities,
        source_nodes={0: "A", 1: "B"},
        sink_nodes={0: (0, "B"), 1: (0, 1)},
        source_to_sinks={0: (0,), 1: (1,)},
        language_shape={"num_cores": 1, "max_num_states": 1},
    )
    core = compile_language_core(fixture, device)
    first = core.operand_table[core.semantic_sink_to_runtime[0], :2].tolist()
    second = core.operand_table[core.semantic_sink_to_runtime[1], :2].tolist()
    terminal_zero_source_first = first[1]
    terminal_zero_source_second = second[1]
    assert terminal_zero_source_first == terminal_zero_source_second
