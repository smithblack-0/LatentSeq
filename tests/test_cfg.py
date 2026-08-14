"""CFG tests certify exact counts, iterative connectivity guarantees, metadata, and feasibility checks."""

import numpy as np
import pytest

from latentseq.language.cfg import _choose_create_or_reuse, parse_cfg, sample_cfg


def _classify(rhs: tuple[object, ...]) -> str:
    ints = [isinstance(symbol, int) for symbol in rhs]
    if len(rhs) == 2 and all(ints):
        return "terminal_pair"
    if len(rhs) == 3 and ints == [True, False, True]:
        return "parenthesis"
    if len(rhs) == 2 and ints in ([True, False], [False, True]):
        return "iteration"
    if len(rhs) == 2 and ints == [False, False]:
        return "branch"
    raise AssertionError(rhs)


def _productive_and_reachable(parsed):
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
    return productive, reachable


def test_sample_cfg_exact_counts_and_consistency(language_config):
    config = language_config["grammar"]
    np.random.seed(77)
    grammar = sample_cfg(config)
    parsed = parse_cfg(grammar)
    counts = {name: 0 for name in ["terminal_pair", "parenthesis", "iteration", "branch"]}
    for _, rhs in parsed.productions:
        counts[_classify(rhs)] += 1
    assert counts == {
        "terminal_pair": config["terminal_pair_rules"],
        "parenthesis": config["parenthesis_rules"],
        "iteration": config["iteration_rules"],
        "branch": config["branch_rules"],
    }
    terminals = sorted({symbol for _, rhs in parsed.productions for symbol in rhs if isinstance(symbol, int)})
    assert terminals == list(range(len(terminals)))
    assert len(terminals) <= config["max_terminals"]
    assert len(parsed.source_nodes) <= config["max_nonterminals"]
    assert len(parsed.productions) == len(set(parsed.productions))
    productive, reachable = _productive_and_reachable(parsed)
    sources = set(parsed.source_nodes.values())
    assert productive == sources
    assert reachable == sources


def test_cfg_metadata_round_trips(language_config):
    grammar = sample_cfg(language_config["grammar"])
    parsed = parse_cfg(grammar)
    assert parsed.language_shape == language_config["grammar"]["language_shape"]
    assert parsed.sampling_defaults == language_config["grammar"]["sampling_defaults"]


def test_cfg_reproducible_when_caller_resets_numpy(language_config):
    np.random.seed(55)
    first = sample_cfg(language_config["grammar"])
    np.random.seed(55)
    second = sample_cfg(language_config["grammar"])
    assert first == second


def test_cfg_rejects_unknown_and_each_capacity_failure(language_config):
    config = language_config["grammar"]
    with pytest.raises(ValueError):
        sample_cfg({**config, "mystery": 1})
    with pytest.raises(ValueError, match="terminal_pair"):
        sample_cfg({**config, "terminal_pair_rules": config["max_nonterminals"] * config["max_terminals"] ** 2 + 1})
    with pytest.raises(ValueError, match="RHS capacity"):
        sample_cfg({**config, "terminal_pair_rules": config["max_terminals"] ** 2 + 1, "parenthesis_rules": 0, "iteration_rules": 0, "branch_rules": 0})
    with pytest.raises(ValueError, match="parenthesis"):
        sample_cfg({**config, "parenthesis_rules": config["max_nonterminals"] ** 2 * config["max_terminals"] ** 2 + 1})
    with pytest.raises(ValueError, match="iteration"):
        sample_cfg({**config, "iteration_rules": 2 * config["max_nonterminals"] ** 2 * config["max_terminals"] + 1})
    with pytest.raises(ValueError, match="branch"):
        sample_cfg({**config, "branch_rules": config["max_nonterminals"] ** 3 + 1})


def test_parse_cfg_supports_source_override():
    grammar = """language_shape: num_cores=2, max_num_states=3;
sampling_defaults: pairwise_odds=4, min_ppl=1.1, max_ppl=3;

A -> B;
A -> 0;
B -> 1;
A: ppl=1.5;
"""
    parsed = parse_cfg(grammar)
    assert parsed.source_overrides["A"] == {"ppl": 1.5}
    assert parsed.start_symbol == "A"


def test_binary_create_reuse_choice_is_uniform():
    np.random.seed(700)
    draws = [_choose_create_or_reuse(True, ["A"]) for _ in range(20000)]
    create_fraction = draws.count("create") / len(draws)
    assert 0.48 <= create_fraction <= 0.52


def test_feasible_configuration_matrix_constructs_without_whole_grammar_retry(language_config):
    base = language_config["grammar"]
    variants = [
        {**base, "terminal_pair_rules": 1, "parenthesis_rules": 0, "iteration_rules": 0, "branch_rules": 0},
        {**base, "terminal_pair_rules": 3, "parenthesis_rules": 1, "iteration_rules": 1, "branch_rules": 1},
        {**base, "terminal_pair_rules": 6, "parenthesis_rules": 3, "iteration_rules": 2, "branch_rules": 2},
    ]
    for seed in range(20):
        for config in variants:
            np.random.seed(seed)
            parsed = parse_cfg(sample_cfg(config))
            productive, reachable = _productive_and_reachable(parsed)
            assert productive == reachable == set(parsed.source_nodes.values())


def test_cfg_metadata_does_not_change_structural_sampling(language_config):
    first = language_config["grammar"]
    second = {
        **first,
        "language_shape": {"num_cores": 7, "max_num_states": 11},
        "sampling_defaults": {"pairwise_odds": 20.0, "min_ppl": 1.01, "max_ppl": 4.0},
    }
    np.random.seed(91)
    parsed_first = parse_cfg(sample_cfg(first))
    np.random.seed(91)
    parsed_second = parse_cfg(sample_cfg(second))
    assert parsed_first.productions == parsed_second.productions


def test_choose_existing_is_uniform_over_exactly_eligible_symbols():
    from latentseq.language.cfg import _choose_existing

    np.random.seed(701)
    draws = [_choose_existing(["eligible-a", "eligible-b"]) for _ in range(20000)]
    assert set(draws) == {"eligible-a", "eligible-b"}
    fraction = draws.count("eligible-a") / len(draws)
    assert 0.48 <= fraction <= 0.52
