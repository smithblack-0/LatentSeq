"""CFG construction and parsing provide LatentSeq's human-readable language boundary.

`sample_cfg` implements the iterative Unold-style construction used by the specification: terminal
pair rules establish productive symbols, later rules attach only to productive symbols, hanging
productive symbols are connected before RHS capacity is exhausted, and the most recently created
nonterminal becomes the start symbol. `parse_cfg` lowers the rendered text into semantic indexes used
by Language Sampling without changing grammar meaning.
"""

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from latentseq._validation import require_exact_keys, require_positive_int


# helpers

CFG_FIELDS = {
    "terminal_pair_rules",
    "parenthesis_rules",
    "iteration_rules",
    "branch_rules",
    "max_terminals",
    "max_nonterminals",
    "language_shape",
    "sampling_defaults",
}
LANGUAGE_SHAPE_FIELDS = {"num_cores", "max_num_states"}
SAMPLING_DEFAULT_FIELDS = {"pairwise_odds", "min_ppl", "max_ppl"}
SOURCE_CONTROL_FIELDS = {"pairwise_odds", "ppl", "nats"}

_METADATA_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*;$")
_PRODUCTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*->\s*(.*?)\s*;$")


def _number(value: str) -> int | float:
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"invalid numeric metadata value {value!r}") from error


def _parse_assignment_list(text: str) -> dict[str, int | float]:
    if not text.strip():
        return {}
    result: dict[str, int | float] = {}
    for assignment in text.split(","):
        if "=" not in assignment:
            raise ValueError(f"invalid metadata assignment {assignment!r}")
        key, raw_value = assignment.split("=", 1)
        key = key.strip()
        if key in result:
            raise ValueError(f"duplicate metadata key {key!r}")
        result[key] = _number(raw_value.strip())
    return result


def _validate_language_shape(shape: object) -> dict[str, int]:
    if not isinstance(shape, dict):
        raise ValueError("language_shape must be a mapping")
    require_exact_keys(shape, LANGUAGE_SHAPE_FIELDS, "language_shape")
    return {
        "num_cores": require_positive_int(shape["num_cores"], "num_cores"),
        "max_num_states": require_positive_int(
            shape["max_num_states"], "max_num_states"
        ),
    }


def _validate_sampling_defaults(defaults: object) -> dict[str, float]:
    if not isinstance(defaults, dict):
        raise ValueError("sampling_defaults must be a mapping")
    unknown = set(defaults) - SAMPLING_DEFAULT_FIELDS
    if unknown:
        raise ValueError(f"unknown sampling_defaults fields: {sorted(unknown)}")
    if "pairwise_odds" not in defaults:
        raise ValueError("sampling_defaults requires pairwise_odds")
    pairwise = defaults["pairwise_odds"]
    if isinstance(pairwise, bool) or not isinstance(pairwise, (int, float)) or pairwise < 1:
        raise ValueError("pairwise_odds must be >= 1")
    result = {"pairwise_odds": float(pairwise)}
    for name in ("min_ppl", "max_ppl"):
        if name in defaults:
            value = defaults[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 1:
                raise ValueError(f"{name} must be > 1")
            result[name] = float(value)
    if "min_ppl" in result and "max_ppl" in result:
        if result["min_ppl"] > result["max_ppl"]:
            raise ValueError("min_ppl cannot exceed max_ppl")
    return result


def _validate_cfg_config(config: dict[str, object]) -> dict[str, object]:
    require_exact_keys(config, CFG_FIELDS, "CFG configuration")
    counts: dict[str, int] = {}
    for name in (
        "terminal_pair_rules",
        "parenthesis_rules",
        "iteration_rules",
        "branch_rules",
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        counts[name] = value
    if counts["terminal_pair_rules"] <= 0:
        raise ValueError("terminal_pair_rules must be positive")
    max_terminals = require_positive_int(config["max_terminals"], "max_terminals")
    max_nonterminals = require_positive_int(
        config["max_nonterminals"], "max_nonterminals"
    )
    language_shape = _validate_language_shape(config["language_shape"])
    sampling_defaults = _validate_sampling_defaults(config["sampling_defaults"])

    terminal_pair = counts["terminal_pair_rules"]
    parenthesis = counts["parenthesis_rules"]
    iteration = counts["iteration_rules"]
    branch = counts["branch_rules"]
    t = max_terminals
    n = max_nonterminals
    if terminal_pair > n * t**2:
        raise ValueError("terminal_pair_rules exceed available rule capacity")
    minimum_productive_sources = math.ceil(terminal_pair / t**2)
    if minimum_productive_sources > parenthesis + iteration + 2 * branch + 1:
        raise ValueError("insufficient later RHS capacity to connect productive sources")
    if parenthesis > n**2 * t**2:
        raise ValueError("parenthesis_rules exceed available rule capacity")
    if iteration > 2 * n**2 * t:
        raise ValueError("iteration_rules exceed available rule capacity")
    if branch > n**3:
        raise ValueError("branch_rules exceed available rule capacity")

    return {
        **counts,
        "max_terminals": max_terminals,
        "max_nonterminals": max_nonterminals,
        "language_shape": language_shape,
        "sampling_defaults": sampling_defaults,
    }


def _nonterminal_name(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return f"N{index}"


def _choose_existing(values: list[int | str]) -> int | str:
    if not values:
        raise ValueError("cannot choose from an empty symbol set")
    return values[int(np.random.randint(0, len(values)))]


def _choose_create_or_reuse(can_create: bool, existing: list[int | str]) -> str:
    """Choose uniformly between creation and reuse when both actions are available."""
    can_reuse = bool(existing)
    if can_create and can_reuse:
        return "create" if int(np.random.randint(0, 2)) == 0 else "reuse"
    if can_create:
        return "create"
    if can_reuse:
        return "reuse"
    raise ValueError("no legal create/reuse action remains")


def _rhs_capacity(remaining: dict[str, int]) -> int:
    return (
        remaining["parenthesis"]
        + remaining["iteration"]
        + 2 * remaining["branch"]
    )


def _render_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return format(value, ".12g")


def _render_cfg(
    productions: list[tuple[str, tuple[int | str, ...]]],
    start_symbol: str,
    language_shape: dict[str, int],
    sampling_defaults: dict[str, float],
) -> str:
    start_rules = [rule for rule in productions if rule[0] == start_symbol]
    other_rules = [rule for rule in productions if rule[0] != start_symbol]
    ordered = start_rules + other_rules
    defaults = ", ".join(
        f"{key}={_render_number(value)}" for key, value in sampling_defaults.items()
    )
    lines = [
        "language_shape: "
        f"num_cores={language_shape['num_cores']}, "
        f"max_num_states={language_shape['max_num_states']};",
        f"sampling_defaults: {defaults};",
        "",
    ]
    for lhs, rhs in ordered:
        lines.append(f"{lhs} -> {' '.join(str(symbol) for symbol in rhs)};")
    return "\n".join(lines) + "\n"


def _collect_nonterminal_children(
    rhs: Iterable[int | str],
) -> set[str]:
    return {symbol for symbol in rhs if isinstance(symbol, str)}


# main


@dataclass(slots=True)
class ParsedCFG:
    """Represent parsed CFG topology and metadata for semantic language construction."""

    grammar: str
    language_shape: dict[str, int]
    sampling_defaults: dict[str, float]
    source_overrides: dict[str, dict[str, float]]
    productions: list[tuple[str, tuple[int | str, ...]]]
    start_symbol: str
    source_nodes: dict[int, str]
    sink_nodes: dict[int, tuple[int | str, ...]]
    source_to_sinks: dict[int, tuple[int, ...]]


def parse_cfg(grammar: str) -> ParsedCFG:
    """Parse one LatentSeq CFG text into semantic source and sink indexes.

    Args:
        grammar: CFG text containing `language_shape`, `sampling_defaults`, productions, and
            optional per-source sampling-control overrides.

    Returns:
        Parsed semantic representation. Source order follows first LHS appearance; sink order
        follows production appearance.
    """
    if not isinstance(grammar, str) or not grammar.strip():
        raise ValueError("grammar must be a nonempty string")

    language_shape: dict[str, int] | None = None
    sampling_defaults: dict[str, float] | None = None
    source_overrides: dict[str, dict[str, float]] = {}
    productions: list[tuple[str, tuple[int | str, ...]]] = []

    for raw_line in grammar.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        production_match = _PRODUCTION_RE.match(line)
        if production_match:
            lhs, rhs_text = production_match.groups()
            rhs_tokens = rhs_text.split()
            if not rhs_tokens:
                raise ValueError("epsilon productions are not supported")
            rhs: list[int | str] = []
            for token in rhs_tokens:
                if re.fullmatch(r"\d+", token):
                    rhs.append(int(token))
                elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
                    rhs.append(token)
                else:
                    raise ValueError(f"invalid CFG symbol {token!r}")
            rule = (lhs, tuple(rhs))
            if rule in productions:
                raise ValueError(f"duplicate CFG production {line!r}")
            productions.append(rule)
            continue

        metadata_match = _METADATA_RE.match(line)
        if not metadata_match:
            raise ValueError(f"cannot parse CFG line {line!r}")
        name, assignments_text = metadata_match.groups()
        assignments = _parse_assignment_list(assignments_text)
        if name == "language_shape":
            if language_shape is not None:
                raise ValueError("duplicate language_shape metadata")
            language_shape = _validate_language_shape(assignments)
        elif name == "sampling_defaults":
            if sampling_defaults is not None:
                raise ValueError("duplicate sampling_defaults metadata")
            sampling_defaults = _validate_sampling_defaults(assignments)
        else:
            if name in source_overrides:
                raise ValueError(f"duplicate override for source {name!r}")
            unknown = set(assignments) - SOURCE_CONTROL_FIELDS
            if unknown or len(assignments) != 1:
                raise ValueError(
                    f"source override {name!r} must contain exactly one of "
                    "pairwise_odds, ppl, nats"
                )
            value = next(iter(assignments.values()))
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("source sampling control must be numeric")
            source_overrides[name] = {
                next(iter(assignments.keys())): float(value)
            }

    if language_shape is None or sampling_defaults is None:
        raise ValueError("grammar requires language_shape and sampling_defaults metadata")
    if not productions:
        raise ValueError("grammar requires at least one production")

    source_order: list[str] = []
    for lhs, _ in productions:
        if lhs not in source_order:
            source_order.append(lhs)
    source_set = set(source_order)
    for source in source_overrides:
        if source not in source_set:
            raise ValueError(f"sampling override references unknown source {source!r}")
    for _, rhs in productions:
        for symbol in rhs:
            if isinstance(symbol, str) and symbol not in source_set:
                raise ValueError(f"RHS references undefined nonterminal {symbol!r}")

    source_nodes = {index: source for index, source in enumerate(source_order)}
    source_index = {source: index for index, source in source_nodes.items()}
    sink_nodes = {index: rhs for index, (_, rhs) in enumerate(productions)}
    grouped: dict[int, list[int]] = {index: [] for index in source_nodes}
    for sink_index, (lhs, _) in enumerate(productions):
        grouped[source_index[lhs]].append(sink_index)
    source_to_sinks = {index: tuple(sinks) for index, sinks in grouped.items()}

    return ParsedCFG(
        grammar=grammar,
        language_shape=language_shape,
        sampling_defaults=sampling_defaults,
        source_overrides=source_overrides,
        productions=productions,
        start_symbol=productions[0][0],
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
        source_to_sinks=source_to_sinks,
    )


def sample_cfg(
    cfg_configuration: dict[str, object],
    _parse_cfg=parse_cfg,
) -> str:
    """Sample one reachable, productive CFG using the documented iterative construction.

    Args:
        cfg_configuration: Exact rule-count, symbol-limit, and Language-metadata mapping.

    Returns:
        Human-readable CFG text with exact requested rule counts and dense terminal IDs.
    """
    config = _validate_cfg_config(cfg_configuration)
    max_terminals = int(config["max_terminals"])
    max_nonterminals = int(config["max_nonterminals"])

    terminals: list[int] = []
    nonterminals: list[str] = []
    productions: list[tuple[str, tuple[int | str, ...]]] = []

    def create_terminal() -> int:
        if len(terminals) >= max_terminals:
            raise RuntimeError("CFG construction exhausted terminal capacity")
        terminal = len(terminals)
        terminals.append(terminal)
        return terminal

    def choose_terminal() -> int:
        action = _choose_create_or_reuse(
            len(terminals) < max_terminals, list(terminals)
        )
        return create_terminal() if action == "create" else int(_choose_existing(list(terminals)))

    def create_nonterminal() -> str:
        if len(nonterminals) >= max_nonterminals:
            raise RuntimeError("CFG construction exhausted nonterminal capacity")
        symbol = _nonterminal_name(len(nonterminals))
        nonterminals.append(symbol)
        return symbol

    # Productive terminal-pair phase. New/reused symbol choices are decision-local rather than
    # sampling uniformly from the space of complete productions.
    terminal_pair_target = int(config["terminal_pair_rules"])
    attempts = 0
    while sum(
        1
        for _, rhs in productions
        if len(rhs) == 2 and all(isinstance(symbol, int) for symbol in rhs)
    ) < terminal_pair_target:
        attempts += 1
        if attempts > terminal_pair_target * 10000:
            raise RuntimeError("failed to construct a unique terminal-pair rule")

        # Creating enough LHS symbols to fit remaining unique terminal pairs is mandatory.
        current_terminal_rules = [
            rule
            for rule in productions
            if len(rule[1]) == 2 and all(isinstance(symbol, int) for symbol in rule[1])
        ]
        remaining_including_current = terminal_pair_target - len(current_terminal_rules)
        existing_capacity = sum(
            max_terminals**2
            - sum(1 for lhs, _ in current_terminal_rules if lhs == source)
            for source in nonterminals
        )
        must_create_source = remaining_including_current > existing_capacity
        maximum_initial_sources = (
            int(config["parenthesis_rules"])
            + int(config["iteration_rules"])
            + 2 * int(config["branch_rules"])
            + 1
        )
        can_create_source = (
            len(nonterminals) < max_nonterminals
            and len(nonterminals) < maximum_initial_sources
        )
        if not nonterminals or must_create_source:
            if not can_create_source:
                raise RuntimeError("CFG construction cannot place remaining terminal-pair rules")
            lhs = create_nonterminal()
        else:
            source_action = _choose_create_or_reuse(can_create_source, list(nonterminals))
            lhs = (
                create_nonterminal()
                if source_action == "create"
                else str(_choose_existing(list(nonterminals)))
            )

        first_terminal = create_terminal() if not terminals else choose_terminal()
        second_terminal = choose_terminal()
        candidate = (lhs, (first_terminal, second_terminal))
        if candidate in productions:
            continue
        productions.append(candidate)

    latest_productive = nonterminals[-1]
    connected: set[str] = {latest_productive}
    hanging: set[str] = set(nonterminals[:-1])

    remaining = {
        "parenthesis": int(config["parenthesis_rules"]),
        "iteration": int(config["iteration_rules"]),
        "branch": int(config["branch_rules"]),
    }

    def connect_rhs_children(lhs: str, rhs: tuple[int | str, ...]) -> None:
        nonlocal connected, hanging
        if lhs not in connected:
            return
        newly_connected = _collect_nonterminal_children(rhs) & hanging
        if newly_connected:
            connected |= newly_connected
            hanging -= newly_connected

    while any(remaining.values()):
        available_types = [name for name, count in remaining.items() if count > 0]
        rule_type = str(_choose_existing(available_types))
        rhs_slots = 2 if rule_type == "branch" else 1
        remaining_after = dict(remaining)
        remaining_after[rule_type] -= 1
        future_capacity = _rhs_capacity(remaining_after)
        required_hanging = max(0, len(hanging) - future_capacity)
        if required_hanging > rhs_slots:
            raise RuntimeError("CFG construction reached an impossible hanging-symbol state")

        attempts = 0
        while True:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("failed to construct a unique nonterminal-bearing rule")

            can_create_lhs = len(nonterminals) < max_nonterminals
            # A new LHS must spend one RHS slot on the previous latest symbol. Do not take that
            # action when the current rule must instead spend all available slots connecting hangs.
            new_lhs_connect_capacity = rhs_slots - 1
            create_preserves_connectivity = (
                len(hanging) - max(0, new_lhs_connect_capacity) <= future_capacity
            )
            can_create_lhs = can_create_lhs and create_preserves_connectivity

            existing_choices = list(nonterminals)
            if required_hanging:
                existing_choices = [source for source in nonterminals if source in connected]
            lhs_action = _choose_create_or_reuse(can_create_lhs, existing_choices)
            previous_latest = latest_productive
            if lhs_action == "create":
                lhs = create_nonterminal()
                latest_productive = lhs
                connected.add(lhs)
                forced_rhs_nonterminals: list[str] = [previous_latest]
            else:
                lhs = str(_choose_existing(existing_choices))
                forced_rhs_nonterminals = []

            available_rhs_slots = rhs_slots - len(forced_rhs_nonterminals)
            forced_hanging_count = max(
                required_hanging,
                max(0, len(hanging) - future_capacity),
            )
            forced_hanging_count = min(forced_hanging_count, available_rhs_slots)
            if forced_hanging_count:
                hanging_choices = list(hanging)
                for _ in range(forced_hanging_count):
                    selected = str(_choose_existing(hanging_choices))
                    forced_rhs_nonterminals.append(selected)
                    hanging_choices.remove(selected)

            rhs_nonterminals = list(forced_rhs_nonterminals)
            while len(rhs_nonterminals) < rhs_slots:
                rhs_nonterminals.append(str(_choose_existing(list(nonterminals))))

            # New LHS symbols cannot appear on the same RHS that made them productive. A rejected
            # local candidate must also undo the local symbol creation before it is redrawn.
            if lhs_action == "create" and lhs in rhs_nonterminals:
                nonterminals.pop()
                connected.discard(lhs)
                latest_productive = previous_latest
                continue
            np.random.shuffle(rhs_nonterminals)

            if rule_type == "parenthesis":
                rhs = (choose_terminal(), rhs_nonterminals[0], choose_terminal())
            elif rule_type == "iteration":
                terminal = choose_terminal()
                if int(np.random.randint(0, 2)) == 0:
                    rhs = (terminal, rhs_nonterminals[0])
                else:
                    rhs = (rhs_nonterminals[0], terminal)
            else:
                rhs = (rhs_nonterminals[0], rhs_nonterminals[1])

            candidate = (lhs, rhs)
            if candidate in productions:
                # Redraw only this rule. If the candidate created a fresh LHS, undo that local
                # creation before retrying so a duplicate cannot leak construction state.
                if lhs_action == "create":
                    nonterminals.pop()
                    connected.discard(lhs)
                    latest_productive = previous_latest
                continue
            productions.append(candidate)
            connect_rhs_children(lhs, rhs)
            break

        remaining[rule_type] -= 1

    if hanging:
        raise RuntimeError("CFG construction left productive symbols unreachable")

    start_symbol = latest_productive
    grammar = _render_cfg(
        productions,
        start_symbol,
        config["language_shape"],
        config["sampling_defaults"],
    )

    # This is an implementation audit, not a retry path: a feasible configuration reaching an
    # invalid final grammar is a construction defect.
    parsed = _parse_cfg(grammar)
    productive: set[str] = set()
    changed = True
    while changed:
        changed = False
        for lhs, rhs in parsed.productions:
            if all(isinstance(symbol, int) or symbol in productive for symbol in rhs):
                if lhs not in productive:
                    productive.add(lhs)
                    changed = True
    reachable = {parsed.start_symbol}
    changed = True
    while changed:
        changed = False
        for lhs, rhs in parsed.productions:
            if lhs in reachable:
                for symbol in rhs:
                    if isinstance(symbol, str) and symbol not in reachable:
                        reachable.add(symbol)
                        changed = True
    sources = set(parsed.source_nodes.values())
    if productive != sources or reachable != sources:
        raise RuntimeError("CFG construction violated productivity/reachability invariants")
    return grammar
