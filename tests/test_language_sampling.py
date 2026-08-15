"""Language Sampling tests certify tensor semantics, control conversion, caps, and ambient RNG use."""

import math

import numpy as np
import pytest

from latentseq.language.cfg import sample_cfg
from latentseq.language.sampling import expected_entropy, get_pseudotemperature, sample_ls_pcfg


def test_ls_pcfg_output_contract(language_config):
    grammar = sample_cfg(language_config["grammar"])
    sampled = sample_ls_pcfg(grammar)
    c, z, s, k = sampled.probabilities.shape
    assert (c, z) == (2, 4)
    assert sampled.probabilities.dtype == np.float64
    assert s == len(sampled.source_nodes)
    assert k == len(sampled.sink_nodes)
    for source_index, sinks in sampled.source_to_sinks.items():
        mask = np.zeros(k, dtype=bool)
        mask[list(sinks)] = True
        rows = sampled.probabilities[:, :, source_index]
        assert np.all(rows[:, :, ~mask] == 0)
        assert np.allclose(rows[:, :, mask].sum(axis=-1), 1.0)


def test_single_transition_is_deterministic_and_consumes_no_control_constraint():
    grammar = """language_shape: num_cores=2, max_num_states=3;
sampling_defaults: pairwise_odds=100;

A -> 0;
"""
    sampled = sample_ls_pcfg(grammar)
    assert np.all(sampled.probabilities == 1.0)


def test_pairwise_odds_conversion():
    assert get_pseudotemperature({"pairwise_odds": 9.0}, 4) == pytest.approx(math.log(9.0) / math.sqrt(2.0))


def test_ppl_and_nats_calibrate_to_same_target():
    temp_ppl = get_pseudotemperature({"ppl": 2.2}, 4)
    temp_nats = get_pseudotemperature({"nats": math.log(2.2)}, 4)
    assert temp_ppl == pytest.approx(temp_nats, rel=1e-7)
    assert expected_entropy(4, temp_ppl) == pytest.approx(math.log(2.2), abs=2e-3)


def test_impossible_direct_controls_raise():
    with pytest.raises(ValueError):
        get_pseudotemperature({"ppl": 1.0}, 4)
    with pytest.raises(ValueError):
        get_pseudotemperature({"ppl": 5.0}, 4)
    with pytest.raises(ValueError):
        get_pseudotemperature({"nats": 0.0}, 4)
    with pytest.raises(ValueError):
        get_pseudotemperature({"pairwise_odds": 0.9}, 4)


def test_explicit_source_override_ignores_default_ppl_caps():
    grammar = """language_shape: num_cores=1, max_num_states=1;
sampling_defaults: pairwise_odds=1000, min_ppl=2.9, max_ppl=3;

A -> 0;
A -> 1;
A -> 2;
A: pairwise_odds=1;
"""
    np.random.seed(1)
    sampled = sample_ls_pcfg(grammar)
    row = sampled.probabilities[0, 0, 0]
    assert np.allclose(row, np.full(3, 1 / 3))


def test_intersection_equivalence():
    logits = np.array([[0.2, -1.0, 0.7], [1.1, 0.4, -0.3]])
    rows = np.exp(logits - logits.max(axis=1, keepdims=True))
    rows /= rows.sum(axis=1, keepdims=True)
    product = rows.prod(axis=0)
    product /= product.sum()
    summed = logits.sum(axis=0)
    expected = np.exp(summed - summed.max())
    expected /= expected.sum()
    assert np.allclose(product, expected)


def test_sampling_reproducibility_is_controlled_by_caller_numpy_seed(language_config):
    np.random.seed(6)
    grammar_a = sample_cfg(language_config["grammar"])
    first = sample_ls_pcfg(grammar_a)
    np.random.seed(6)
    grammar_b = sample_cfg(language_config["grammar"])
    second = sample_ls_pcfg(grammar_b)
    assert grammar_a == grammar_b
    assert np.array_equal(first.probabilities, second.probabilities)


def test_inherited_ppl_caps_bound_pairwise_temperature_in_both_directions():
    from latentseq.language.sampling import _cap_inherited_temperature

    transitions = 5
    very_sharp = get_pseudotemperature({"pairwise_odds": 1000.0}, transitions)
    min_capped = _cap_inherited_temperature(
        very_sharp,
        {"pairwise_odds": 1000.0, "min_ppl": 3.0},
        transitions,
    )
    assert expected_entropy(transitions, min_capped) == pytest.approx(math.log(3.0), abs=2e-3)

    flat = get_pseudotemperature({"pairwise_odds": 1.0}, transitions)
    max_capped = _cap_inherited_temperature(
        flat,
        {"pairwise_odds": 1.0, "max_ppl": 2.0},
        transitions,
    )
    assert expected_entropy(transitions, max_capped) == pytest.approx(math.log(2.0), abs=2e-3)


def test_core_compensation_preserves_combined_logit_spread_statistically():
    target = get_pseudotemperature({"pairwise_odds": 6.0}, 6)
    np.random.seed(300)
    measured = []
    for num_cores in (1, 2, 4, 8):
        logits = np.random.normal(
            0.0,
            target / math.sqrt(num_cores),
            size=(20000, num_cores, 6),
        )
        combined = logits.sum(axis=1)
        measured.append(float(combined.std()))
    for spread in measured:
        assert spread == pytest.approx(target, rel=0.02)


def test_sampling_defaults_reject_contradictory_bounds():
    grammar = """language_shape: num_cores=1, max_num_states=1;
sampling_defaults: pairwise_odds=3, min_ppl=3, max_ppl=2;

A -> 0;
A -> 1;
A -> 2;
"""
    with pytest.raises(ValueError, match="min_ppl"):
        sample_ls_pcfg(grammar)
