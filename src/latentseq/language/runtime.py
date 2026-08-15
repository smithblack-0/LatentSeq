"""Torch runtime components execute one compiled LanguageCore against a Latent Sample.

The runtime is a batch-parallel fixed-shape pushdown protocol. Stack, Reader, Recorder, and
CounterDecoder are per-attempt state; Transitions and InstructionDecoder interpret reusable compiled
tables. `decode_chunk` executes a fixed number of expansion cycles and contains no host-side
completion or overflow decisions, making the chunk the intended `torch.compile` boundary.
"""

from dataclasses import dataclass
from typing import Callable

import torch

from .._sampling_ops import sample_categorical

from latentseq._validation import require_positive_int

from .compilation import OUTPUT, PUSH
from .ir import LanguageCore


# helpers

Instruction = dict[str, torch.Tensor]


def _validate_latent_samples(latent_samples: torch.Tensor) -> None:
    if latent_samples.ndim != 3 or latent_samples.dtype != torch.int64:
        raise ValueError("latent_samples must be int64 [batch,core,timestep]")
    if latent_samples.shape[0] <= 0 or latent_samples.shape[1] <= 0 or latent_samples.shape[2] <= 0:
        raise ValueError("latent_samples dimensions must all be positive")


# main


class Stack:
    """Store pending grammar sources and retain per-lane overflow through a decode chunk.

    Logical capacity is `stack_depth`; physical storage adds one operand-width scratch region so a
    fixed-shape speculative push remains memory-safe before the logical capacity decision.
    """

    def __init__(
        self,
        batch_size: int,
        stack_depth: int,
        operand_size: int,
        start_source_node: int,
        done_source_node: int,
        device: torch.device,
    ) -> None:
        self.stack_depth = stack_depth
        self.operand_size = operand_size
        self.done_source_node = done_source_node
        self.device = device
        self.stack = torch.zeros(
            (batch_size, stack_depth + operand_size),
            dtype=torch.int64,
            device=device,
        )
        self.stack[:, 0] = start_source_node
        self.stack_position = torch.ones(batch_size, dtype=torch.int64, device=device)
        self.overflow = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._batch_index = torch.arange(batch_size, device=device)
        self._operand_offsets = torch.arange(operand_size, device=device)

    def pop(self, partial_instruction: Instruction) -> torch.Tensor:
        """Pop one source for active lanes and route done/overflowed lanes to the done source."""
        inactive = partial_instruction["is_done"] | self.overflow
        active = ~inactive
        candidate_position = torch.clamp(self.stack_position - 1, min=0)
        self.stack_position = torch.where(
            active, candidate_position, self.stack_position
        )
        selected = self.stack[self._batch_index, candidate_position]
        done = torch.full_like(selected, self.done_source_node)
        return torch.where(active, selected, done)

    def push(self, instruction: Instruction) -> None:
        """Speculatively transfer operands, then commit only legal active stack pushes."""
        operand = instruction["operand"]
        inactive = instruction["is_done"] | self.overflow
        copy_mask = ~inactive
        write_indexes = self.stack_position[:, None] + self._operand_offsets[None, :]
        write_indexes = torch.clamp(write_indexes, max=self.stack.shape[1] - 1)
        current = self.stack[self._batch_index[:, None], write_indexes]
        values = torch.where(copy_mask[:, None], operand, current)
        self.stack[self._batch_index[:, None], write_indexes] = values

        is_push = instruction["operator"] == PUSH
        proposed = self.stack_position + instruction["operand_length"]
        overflow_now = is_push & ~inactive & (proposed > self.stack_depth)
        self.overflow = self.overflow | overflow_now
        commit = is_push & ~inactive & ~overflow_now
        self.stack_position = torch.where(commit, proposed, self.stack_position)


class Reader:
    """Read aligned latent states without owning or advancing the emitted-token position."""

    def __init__(self, latent_samples: torch.Tensor) -> None:
        self.latent_samples = latent_samples
        self._batch_index = torch.arange(
            latent_samples.shape[0], device=latent_samples.device
        )

    def __call__(self, partial_instruction: Instruction) -> torch.Tensor:
        """Return int64 `[batch,core]` latent state selected by each lane's safe data counter."""
        return self.latent_samples[
            self._batch_index, :, partial_instruction["data_counter"]
        ]


class Recorder:
    """Own fixed Grammar Sample memory and use same-position speculative writes between outputs."""

    def __init__(
        self,
        batch_size: int,
        num_timesteps: int,
        device: torch.device,
    ) -> None:
        self.grammar_samples = torch.zeros(
            (batch_size, num_timesteps), dtype=torch.int64, device=device
        )
        self._batch_index = torch.arange(batch_size, device=device)

    def record(self, instruction: Instruction) -> None:
        """Write operand zero at the current position while preserving completed lanes."""
        counter = instruction["data_counter"]
        previous = self.grammar_samples[self._batch_index, counter]
        values = torch.where(
            instruction["is_done"], previous, instruction["operand"][:, 0]
        )
        self.grammar_samples[self._batch_index, counter] = values

    def retrieve(self) -> torch.Tensor:
        """Return the fixed Grammar Sample memory."""
        return self.grammar_samples


class CounterDecoder:
    """Own emitted-token position so grammar expansions and token emission remain distinct."""

    def __init__(
        self,
        batch_size: int,
        num_timesteps: int,
        device: torch.device,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.latent_position = torch.zeros(
            batch_size, dtype=torch.int64, device=device
        )

    def decode(self) -> Instruction:
        """Return safe current data indexes plus each lane's completion mask."""
        is_done = self.latent_position >= self.num_timesteps
        data_counter = torch.clamp(self.latent_position, max=self.num_timesteps - 1)
        return {"data_counter": data_counter, "is_done": is_done}

    def step(self, instruction: Instruction) -> None:
        """Advance only active output instructions by one emitted position."""
        advance = (instruction["operator"] == OUTPUT) & ~instruction["is_done"]
        self.latent_position = self.latent_position + advance.to(torch.int64)

    def is_complete(self) -> torch.Tensor:
        """Return a bool completion flag for every batch lane."""
        return self.latent_position >= self.num_timesteps


class Transitions:
    """Apply latent factor conditioning to compiled source/state transition contributions."""

    def __init__(self, transition_tables: torch.Tensor) -> None:
        self.transition_tables = transition_tables
        self.num_cores = transition_tables.shape[0]

    def __call__(
        self,
        source_nodes: torch.Tensor,
        latent_state: torch.Tensor,
    ) -> torch.Tensor:
        """Sample one runtime sink for each batch lane from the normalized core product."""
        batch_size = source_nodes.shape[0]
        core_index = torch.arange(
            self.num_cores, device=source_nodes.device
        )[None, :].expand(batch_size, self.num_cores)
        source_index = source_nodes[:, None].expand(batch_size, self.num_cores)
        raw = self.transition_tables[
            core_index, latent_state, source_index, :
        ]
        valid = torch.all(raw > 0, dim=1)
        safe_raw = torch.where(raw > 0, raw, torch.ones_like(raw))
        log_product = torch.log(safe_raw).sum(dim=1)
        negative_infinity = torch.full_like(log_product, float("-inf"))
        logits = torch.where(valid, log_product, negative_infinity)
        probabilities = torch.softmax(logits, dim=-1)
        return sample_categorical(probabilities)


class InstructionDecoder:
    """Map sampled runtime sinks onto the fixed instruction tables compiled for a Language."""

    def __init__(
        self,
        operand_table: torch.Tensor,
        operation_table: torch.Tensor,
        operand_length_table: torch.Tensor,
    ) -> None:
        self.operand_table = operand_table
        self.operation_table = operation_table
        self.operand_length_table = operand_length_table

    def __call__(
        self,
        sink_nodes: torch.Tensor,
        partial_instruction: Instruction,
    ) -> Instruction:
        """Return a full instruction while preserving partial counter/completion fields exactly."""
        return {
            "operand": self.operand_table[sink_nodes],
            "operator": self.operation_table[sink_nodes],
            "operand_length": self.operand_length_table[sink_nodes],
            "data_counter": partial_instruction["data_counter"],
            "is_done": partial_instruction["is_done"],
        }


@dataclass(slots=True)
class RuntimeAttempt:
    """Group the four mutable helpers whose lifetime is exactly one Language generation attempt."""

    stack: Stack
    reader: Reader
    recorder: Recorder
    counter: CounterDecoder


def decode_chunk(
    attempt: RuntimeAttempt,
    transitions: Transitions,
    instruction_decoder: InstructionDecoder,
    chunk_size: int,
) -> None:
    """Execute exactly `chunk_size` fixed-shape grammar-expansion cycles.

    Host-side completion and overflow inspection intentionally occurs outside this function.
    """
    for _ in range(chunk_size):
        partial = attempt.counter.decode()
        source = attempt.stack.pop(partial)
        latent_state = attempt.reader(partial)
        sink = transitions(source, latent_state)
        instruction = instruction_decoder(sink, partial)
        attempt.stack.push(instruction)
        attempt.recorder.record(instruction)
        attempt.counter.step(instruction)


def select_decode_chunk_backend(
    device: torch.device,
    _decode_chunk=decode_chunk,
):
    """Select eager CPU execution or the compiled fixed-trip CUDA chunk backend."""
    if device.type == "cuda":
        return torch.compile(_decode_chunk, dynamic=False, fullgraph=True)
    return _decode_chunk


# construction


def build_runtime_attempt(
    latent_samples: torch.Tensor,
    core: LanguageCore,
    stack_depth: int,
    _stack_cls: type[Stack] = Stack,
    _reader_cls: type[Reader] = Reader,
    _recorder_cls: type[Recorder] = Recorder,
    _counter_cls: type[CounterDecoder] = CounterDecoder,
    _attempt_cls: type[RuntimeAttempt] = RuntimeAttempt,
) -> RuntimeAttempt:
    """Construct all ephemeral runtime state for one generation attempt."""
    _validate_latent_samples(latent_samples)
    batch_size, _, num_timesteps = latent_samples.shape
    device = latent_samples.device
    operand_size = core.operand_table.shape[1]
    return _attempt_cls(
        stack=_stack_cls(
            batch_size,
            stack_depth,
            operand_size,
            core.start_source_node,
            core.done_source_node,
            device,
        ),
        reader=_reader_cls(latent_samples),
        recorder=_recorder_cls(batch_size, num_timesteps, device),
        counter=_counter_cls(batch_size, num_timesteps, device),
    )


def build_transitions(
    core: LanguageCore,
    _cls: type[Transitions] = Transitions,
) -> Transitions:
    """Construct the reusable transition interpreter from a compiled LanguageCore."""
    return _cls(core.transition_tables)


def build_instruction_decoder(
    core: LanguageCore,
    _cls: type[InstructionDecoder] = InstructionDecoder,
) -> InstructionDecoder:
    """Construct the reusable instruction decoder from a compiled LanguageCore."""
    return _cls(
        core.operand_table,
        core.operation_table,
        core.operand_length_table,
    )
