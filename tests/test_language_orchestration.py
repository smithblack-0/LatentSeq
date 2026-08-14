"""Language orchestration tests isolate retry/chunk semantics through construction-time injection."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from latentseq.language.api import build_language_from_ls_pcfg
from latentseq.language.ir import LSPCFG


def _ls_pcfg() -> LSPCFG:
    probabilities = np.ones((1, 1, 1, 1), dtype=np.float64)
    return LSPCFG(
        grammar="orchestration",
        probabilities=probabilities,
        source_nodes={0: "A"},
        sink_nodes={0: (0,)},
        source_to_sinks={0: (0,)},
        language_shape={"num_cores": 1, "max_num_states": 1},
    )


class FakeStack:
    def __init__(self):
        self.overflow = torch.tensor([False])


class FakeCounter:
    def __init__(self):
        self.complete = False

    def is_complete(self):
        return torch.tensor([self.complete])


class FakeRecorder:
    def __init__(self, value: int):
        self.value = value

    def retrieve(self):
        return torch.tensor([[self.value]], dtype=torch.int64)


@dataclass
class FakeAttempt:
    attempt_index: int
    stack: FakeStack
    counter: FakeCounter
    recorder: FakeRecorder
    reader: object = None


class AttemptFactory:
    def __init__(self):
        self.created: list[FakeAttempt] = []

    def __call__(self, latent, core, stack_depth):
        attempt = FakeAttempt(
            attempt_index=len(self.created),
            stack=FakeStack(),
            counter=FakeCounter(),
            recorder=FakeRecorder(100 + len(self.created)),
        )
        self.created.append(attempt)
        return attempt


def _language(runtime, factory, chunk):
    return build_language_from_ls_pcfg(
        _ls_pcfg(),
        runtime,
        torch.device("cpu"),
        _attempt_factory=factory,
        _select_decode_chunk_backend=lambda device: chunk,
    )


def test_runtime_configuration_is_exact_and_positive():
    factory = AttemptFactory()
    chunk = lambda *args: None
    for bad in [
        {"chunk_size": 1, "max_attempts": 1},
        {"stack_depth": 1, "max_attempts": 1},
        {"stack_depth": 1, "chunk_size": 1},
        {"stack_depth": 0, "chunk_size": 1, "max_attempts": 1},
        {"stack_depth": 1, "chunk_size": 0, "max_attempts": 1},
        {"stack_depth": 1, "chunk_size": 1, "max_attempts": 0},
    ]:
        with pytest.raises(ValueError):
            _language(bad, factory, chunk)


def test_fresh_attempt_state_and_immediate_completion():
    factory = AttemptFactory()

    def chunk(attempt, transitions, decoder, chunk_size):
        attempt.counter.complete = True

    language = _language(
        {"stack_depth": 4, "chunk_size": 2, "max_attempts": 2}, factory, chunk
    )
    result = language.sample(torch.zeros((1, 1, 1), dtype=torch.int64))
    assert result.tolist() == [[100]]
    assert len(factory.created) == 1


def test_chunk_continues_until_completion():
    factory = AttemptFactory()
    calls = 0

    def chunk(attempt, transitions, decoder, chunk_size):
        nonlocal calls
        calls += 1
        if calls == 2:
            attempt.counter.complete = True

    language = _language(
        {"stack_depth": 4, "chunk_size": 2, "max_attempts": 2}, factory, chunk
    )
    assert language.sample(torch.zeros((1, 1, 1), dtype=torch.int64)).tolist() == [[100]]
    assert calls == 2
    assert len(factory.created) == 1


def test_overflow_discards_attempt_and_reconstructs_all_ephemeral_state():
    factory = AttemptFactory()

    def chunk(attempt, transitions, decoder, chunk_size):
        if attempt.attempt_index == 0:
            attempt.stack.overflow[:] = True
        else:
            attempt.counter.complete = True

    language = _language(
        {"stack_depth": 4, "chunk_size": 2, "max_attempts": 2}, factory, chunk
    )
    assert language.sample(torch.zeros((1, 1, 1), dtype=torch.int64)).tolist() == [[101]]
    assert len(factory.created) == 2
    assert factory.created[0] is not factory.created[1]


def test_retry_exhaustion_raises_without_partial_result():
    factory = AttemptFactory()

    def chunk(attempt, transitions, decoder, chunk_size):
        attempt.stack.overflow[:] = True

    language = _language(
        {"stack_depth": 4, "chunk_size": 2, "max_attempts": 3}, factory, chunk
    )
    with pytest.raises(RuntimeError, match="stack"):
        language.sample(torch.zeros((1, 1, 1), dtype=torch.int64))
    assert len(factory.created) == 3


def test_overflow_precedes_completion_at_same_boundary():
    factory = AttemptFactory()

    def chunk(attempt, transitions, decoder, chunk_size):
        attempt.counter.complete = True
        if attempt.attempt_index == 0:
            attempt.stack.overflow[:] = True

    language = _language(
        {"stack_depth": 4, "chunk_size": 2, "max_attempts": 2}, factory, chunk
    )
    assert language.sample(torch.zeros((1, 1, 1), dtype=torch.int64)).tolist() == [[101]]
    assert len(factory.created) == 2


def test_retry_continues_ambient_rng_stream_without_reseed_or_rewind():
    factory = AttemptFactory()
    observed: list[float] = []

    def chunk(attempt, transitions, decoder, chunk_size):
        observed.append(float(torch.rand(())))
        if attempt.attempt_index == 0:
            attempt.stack.overflow[:] = True
        else:
            attempt.counter.complete = True

    language = _language(
        {"stack_depth": 4, "chunk_size": 2, "max_attempts": 2}, factory, chunk
    )
    torch.manual_seed(111)
    expected = [float(torch.rand(())), float(torch.rand(()))]
    torch.manual_seed(111)
    language.sample(torch.zeros((1, 1, 1), dtype=torch.int64))
    assert observed == expected
