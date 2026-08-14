"""Runtime helper tests certify the fixed-shape pushdown protocol independently of orchestration."""

import torch

from latentseq.language.compilation import NOOP, OUTPUT, PUSH
from latentseq.language.runtime import (
    CounterDecoder,
    InstructionDecoder,
    Reader,
    Recorder,
    Stack,
    Transitions,
)


def _instruction(operand, operator, length, counter=None, done=None):
    operand = torch.tensor(operand, dtype=torch.int64)
    batch = operand.shape[0]
    return {
        "operand": operand,
        "operator": torch.tensor(operator, dtype=torch.int64),
        "operand_length": torch.tensor(length, dtype=torch.int64),
        "data_counter": torch.zeros(batch, dtype=torch.int64)
        if counter is None
        else torch.tensor(counter, dtype=torch.int64),
        "is_done": torch.zeros(batch, dtype=torch.bool)
        if done is None
        else torch.tensor(done, dtype=torch.bool),
    }


def test_stack_initialization_lifo_partial_and_done_behavior():
    stack = Stack(2, 4, 3, start_source_node=90, done_source_node=99, device=torch.device("cpu"))
    assert stack.stack.shape == (2, 7)
    assert stack.stack_position.tolist() == [1, 1]
    assert stack.stack[:, 0].tolist() == [90, 90]

    partial = {"data_counter": torch.zeros(2, dtype=torch.int64), "is_done": torch.tensor([False, False])}
    assert stack.pop(partial).tolist() == [90, 90]
    stack.push(_instruction([[30, 20, 10], [7, 6, 999]], [PUSH, PUSH], [3, 2]))
    assert stack.pop(partial).tolist() == [10, 6]
    assert stack.pop(partial).tolist() == [20, 7]

    done_partial = {"data_counter": torch.zeros(2, dtype=torch.int64), "is_done": torch.tensor([True, False])}
    before = stack.stack_position.clone()
    nodes = stack.pop(done_partial)
    assert nodes[0].item() == 99
    assert stack.stack_position[0].item() == before[0].item()


def test_stack_nonpush_does_not_commit_and_overflow_persists():
    stack = Stack(1, 2, 3, 1, 9, torch.device("cpu"))
    partial = {"data_counter": torch.zeros(1, dtype=torch.int64), "is_done": torch.tensor([False])}
    stack.pop(partial)
    stack.push(_instruction([[77, 88, 99]], [OUTPUT], [1]))
    assert stack.stack_position.item() == 0
    stack.push(_instruction([[3, 2, 1]], [PUSH], [3]))
    assert stack.overflow.item()
    assert stack.stack_position.item() == 0
    # Further activity in the same chunk remains safe and cannot clear overflow.
    stack.pop(partial)
    stack.push(_instruction([[4, 5, 6]], [PUSH], [3]))
    assert stack.overflow.item()


def test_reader_preserves_batch_core_shape_and_lane_positions():
    latent = torch.empty((2, 3, 4), dtype=torch.int64)
    for b in range(2):
        for c in range(3):
            for t in range(4):
                latent[b, c, t] = 100 * b + 10 * c + t
    reader = Reader(latent)
    partial = {"data_counter": torch.tensor([2, 0]), "is_done": torch.tensor([False, False])}
    result = reader(partial)
    assert result.shape == (2, 3)
    assert result.tolist() == [[2, 12, 22], [100, 110, 120]]


def test_counter_and_recorder_commit_only_outputs_and_keep_done_lane_in_bounds():
    counter = CounterDecoder(1, 3, torch.device("cpu"))
    recorder = Recorder(1, 3, torch.device("cpu"))
    for value, operator in [(99, PUSH), (11, OUTPUT), (88, NOOP), (22, OUTPUT), (33, OUTPUT)]:
        partial = counter.decode()
        instruction = _instruction(
            [[value]],
            [operator],
            [1],
            partial["data_counter"].tolist(),
            partial["is_done"].tolist(),
        )
        recorder.record(instruction)
        counter.step(instruction)
    assert recorder.retrieve().tolist() == [[11, 22, 33]]
    assert counter.is_complete().item()
    done = counter.decode()
    assert done["is_done"].item()
    assert done["data_counter"].item() == 2
    before = recorder.retrieve().clone()
    instruction = _instruction([[999]], [OUTPUT], [1], done["data_counter"].tolist(), done["is_done"].tolist())
    recorder.record(instruction)
    counter.step(instruction)
    assert torch.equal(recorder.retrieve(), before)
    assert counter.latent_position.item() == 3


def test_instruction_decoder_preserves_partial_fields():
    decoder = InstructionDecoder(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        torch.tensor([PUSH, OUTPUT], dtype=torch.int64),
        torch.tensor([2, 1], dtype=torch.int64),
    )
    partial = {"data_counter": torch.tensor([7, 8]), "is_done": torch.tensor([False, True])}
    result = decoder(torch.tensor([1, 0]), partial)
    assert result["operand"].tolist() == [[3, 4], [1, 2]]
    assert result["operator"].tolist() == [OUTPUT, PUSH]
    assert torch.equal(result["data_counter"], partial["data_counter"])
    assert torch.equal(result["is_done"], partial["is_done"])


def test_transitions_source_state_core_order_and_factor_conditioning():
    tables = torch.zeros((2, 2, 2, 4), dtype=torch.float32)
    # Source 0, state pair (0,0) intersects only on sink 1.
    tables[0, 0, 0] = torch.tensor([0.5, 0.5, 0.0, 0.0])
    tables[1, 0, 0] = torch.tensor([0.0, 0.5, 0.5, 0.0])
    # Swapping core states selects sink 3 exactly.
    tables[0, 1, 0, 3] = 1.0
    tables[1, 0, 0, 3] = 1.0
    # Source 1 deterministic sink 2 under every state.
    tables[:, :, 1, 2] = 1.0
    transitions = Transitions(tables)
    sinks = transitions(
        torch.tensor([0, 0, 1]),
        torch.tensor([[0, 0], [1, 0], [1, 1]]),
    )
    assert sinks.tolist() == [1, 3, 2]
