"""Language Compilation lowers semantic LS-PCFG choices into one shared Torch runtime numbering.

Compilation preserves every sampled production probability while adding control nodes and reusable
terminal executors required by the pushdown runtime. Transition and instruction tables are built from
the same numbering plan so they cannot silently disagree about node identity.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .ir import LSPCFG, LanguageCore


# helpers

NOOP = 0
PUSH = 1
OUTPUT = 2


@dataclass(slots=True)
class _RuntimeNumbering:
    semantic_source_to_runtime: dict[int, int]
    semantic_sink_to_runtime: dict[int, int]
    terminal_source: dict[int, int]
    terminal_sink: dict[int, int]
    start_source: int
    done_source: int
    start_sink: int
    done_sink: int
    source_count: int
    sink_count: int
    operand_size: int


def _terminal_ids(ls_pcfg: LSPCFG) -> list[int]:
    return sorted(
        {
            symbol
            for rhs in ls_pcfg.sink_nodes.values()
            for symbol in rhs
            if isinstance(symbol, int)
        }
    )


def _validate_terminal_identity(ls_pcfg: LSPCFG) -> int:
    terminals = _terminal_ids(ls_pcfg)
    if not terminals:
        raise ValueError("Language requires at least one semantic terminal")
    expected = list(range(terminals[-1] + 1))
    if terminals != expected:
        raise ValueError("semantic terminal IDs must be dense from zero")
    return len(terminals)


def _build_numbering(ls_pcfg: LSPCFG) -> _RuntimeNumbering:
    semantic_source_to_runtime = {
        source: source for source in sorted(ls_pcfg.source_nodes)
    }
    semantic_sink_to_runtime = {sink: sink for sink in sorted(ls_pcfg.sink_nodes)}

    executor_terminals = sorted(
        {
            symbol
            for rhs in ls_pcfg.sink_nodes.values()
            if len(rhs) > 1
            for symbol in rhs
            if isinstance(symbol, int)
        }
    )
    next_source = len(semantic_source_to_runtime)
    terminal_source: dict[int, int] = {}
    for terminal in executor_terminals:
        terminal_source[terminal] = next_source
        next_source += 1
    start_source = next_source
    done_source = next_source + 1
    source_count = done_source + 1

    next_sink = len(semantic_sink_to_runtime)
    terminal_sink: dict[int, int] = {}
    for terminal in executor_terminals:
        terminal_sink[terminal] = next_sink
        next_sink += 1
    start_sink = next_sink
    done_sink = next_sink + 1
    sink_count = done_sink + 1

    maximum_rhs = max(len(rhs) for rhs in ls_pcfg.sink_nodes.values())
    operand_size = max(2, maximum_rhs)
    return _RuntimeNumbering(
        semantic_source_to_runtime=semantic_source_to_runtime,
        semantic_sink_to_runtime=semantic_sink_to_runtime,
        terminal_source=terminal_source,
        terminal_sink=terminal_sink,
        start_source=start_source,
        done_source=done_source,
        start_sink=start_sink,
        done_sink=done_sink,
        source_count=source_count,
        sink_count=sink_count,
        operand_size=operand_size,
    )


def _map_rhs_symbol(
    symbol: int | str,
    numbering: _RuntimeNumbering,
    source_name_to_index: dict[str, int],
) -> int:
    if isinstance(symbol, int):
        if symbol not in numbering.terminal_source:
            raise ValueError(
                "terminal executor missing for a terminal used in a compound RHS"
            )
        return numbering.terminal_source[symbol]
    return numbering.semantic_source_to_runtime[source_name_to_index[symbol]]


# main


def compile_language_core(
    ls_pcfg: LSPCFG,
    device: str | torch.device,
) -> LanguageCore:
    """Compile one LS-PCFG into static Torch transition and instruction tables.

    Args:
        ls_pcfg: Semantic sampled language with dense terminal IDs.
        device: Target execution device for the compiled tables.

    Returns:
        `LanguageCore` preserving the LS-PCFG probability semantics and adding deterministic start,
        done, and reusable terminal-executor runtime nodes. Compilation performs no random draws.
    """
    torch_device = torch.device(device)
    probabilities = ls_pcfg.probabilities
    if probabilities.ndim != 4 or probabilities.dtype != np.float64:
        raise ValueError("LS-PCFG probabilities must be float64 [core,state,source,sink]")
    if probabilities.shape[2] != len(ls_pcfg.source_nodes):
        raise ValueError("LS-PCFG source mapping does not match probability tensor")
    if probabilities.shape[3] != len(ls_pcfg.sink_nodes):
        raise ValueError("LS-PCFG sink mapping does not match probability tensor")

    grammar_terminal_count = _validate_terminal_identity(ls_pcfg)
    numbering = _build_numbering(ls_pcfg)
    num_cores, max_num_states = probabilities.shape[:2]
    transition_tables = torch.zeros(
        (
            num_cores,
            max_num_states,
            numbering.source_count,
            numbering.sink_count,
        ),
        dtype=torch.float32,
        device=torch_device,
    )

    for semantic_source, valid_sinks in ls_pcfg.source_to_sinks.items():
        runtime_source = numbering.semantic_source_to_runtime[semantic_source]
        for semantic_sink in valid_sinks:
            runtime_sink = numbering.semantic_sink_to_runtime[semantic_sink]
            transition_tables[:, :, runtime_source, runtime_sink] = torch.from_numpy(
                probabilities[:, :, semantic_source, semantic_sink]
            ).to(device=torch_device, dtype=torch.float32)

    for terminal, runtime_source in numbering.terminal_source.items():
        runtime_sink = numbering.terminal_sink[terminal]
        transition_tables[:, :, runtime_source, runtime_sink] = 1.0
    transition_tables[:, :, numbering.start_source, numbering.start_sink] = 1.0
    transition_tables[:, :, numbering.done_source, numbering.done_sink] = 1.0

    operand_table = torch.zeros(
        (numbering.sink_count, numbering.operand_size),
        dtype=torch.int64,
        device=torch_device,
    )
    operation_table = torch.full(
        (numbering.sink_count,), NOOP, dtype=torch.int64, device=torch_device
    )
    operand_length_table = torch.zeros(
        (numbering.sink_count,), dtype=torch.int64, device=torch_device
    )
    source_name_to_index = {
        name: index for index, name in ls_pcfg.source_nodes.items()
    }

    for semantic_sink, rhs in ls_pcfg.sink_nodes.items():
        runtime_sink = numbering.semantic_sink_to_runtime[semantic_sink]
        if len(rhs) == 1 and isinstance(rhs[0], int):
            operation_table[runtime_sink] = OUTPUT
            operand_table[runtime_sink, 0] = rhs[0]
            operand_length_table[runtime_sink] = 1
            continue
        runtime_rhs = [
            _map_rhs_symbol(symbol, numbering, source_name_to_index) for symbol in rhs
        ]
        runtime_rhs.reverse()
        operation_table[runtime_sink] = PUSH
        operand_length_table[runtime_sink] = len(runtime_rhs)
        operand_table[runtime_sink, : len(runtime_rhs)] = torch.tensor(
            runtime_rhs, dtype=torch.int64, device=torch_device
        )

    for terminal, runtime_sink in numbering.terminal_sink.items():
        operation_table[runtime_sink] = OUTPUT
        operand_table[runtime_sink, 0] = terminal
        operand_length_table[runtime_sink] = 1

    grammar_start_source_node = numbering.semantic_source_to_runtime[0]
    operation_table[numbering.start_sink] = PUSH
    operand_table[numbering.start_sink, :2] = torch.tensor(
        [numbering.start_source, grammar_start_source_node],
        dtype=torch.int64,
        device=torch_device,
    )
    operand_length_table[numbering.start_sink] = 2
    operation_table[numbering.done_sink] = NOOP
    operand_length_table[numbering.done_sink] = 0

    return LanguageCore(
        transition_tables=transition_tables,
        operand_table=operand_table,
        operation_table=operation_table,
        operand_length_table=operand_length_table,
        start_source_node=numbering.start_source,
        done_source_node=numbering.done_source,
        grammar_terminal_count=grammar_terminal_count,
        semantic_source_to_runtime=dict(numbering.semantic_source_to_runtime),
        semantic_sink_to_runtime=dict(numbering.semantic_sink_to_runtime),
        grammar_start_source_node=grammar_start_source_node,
    )


__all__ = [
    "NOOP",
    "OUTPUT",
    "PUSH",
    "LanguageCore",
    "compile_language_core",
]
