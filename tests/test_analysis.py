"""Analysis tests protect reduction order, expected multiplicity, finite-horizon semantics, and recurrence."""

import numpy as np

from latentseq.language.analysis import analyze_language
from latentseq.language.ir import LSPCFG


def _fixture() -> LSPCFG:
    probabilities = np.zeros((2, 2, 2, 3), dtype=np.float64)
    probabilities[0, 0, 0, :2] = [0.9, 0.1]
    probabilities[0, 1, 0, :2] = [0.1, 0.9]
    probabilities[1, 0, 0, :2] = [0.8, 0.2]
    probabilities[1, 1, 0, :2] = [0.8, 0.2]
    probabilities[:, :, 1, 2] = 1.0
    return LSPCFG(
        grammar="fixture",
        probabilities=probabilities,
        source_nodes={0: "A", 1: "B"},
        sink_nodes={0: ("B", "B", 0), 1: (1,), 2: (0,)},
        source_to_sinks={0: (0, 1), 1: (2,)},
        language_shape={"num_cores": 2, "max_num_states": 2},
    )


def test_analysis_averages_states_then_intersects():
    fixture = _fixture()
    analysis = analyze_language(fixture, {"trace_depth": 5})
    core_average = fixture.probabilities.mean(axis=1)
    for source in range(2):
        valid = core_average[:, source].sum(axis=0) > 0
        expected = core_average[:, source, valid].prod(axis=0)
        expected /= expected.sum()
        assert np.allclose(analysis.reference_probabilities[source, valid], expected)


def test_analysis_counts_rhs_multiplicity_and_preserves_expected_mass():
    analysis = analyze_language(_fixture(), {"trace_depth": 3})
    assert analysis.nonterminal_offspring[0, 1] > 1.0
    assert analysis.terminal_emission.shape[1] == 2
    assert analysis.max_rhs_length == 3
    assert analysis.source_mass[1].sum() > 1.0


def test_analysis_trace_extension_preserves_prefix():
    short = analyze_language(_fixture(), {"trace_depth": 3})
    long = analyze_language(_fixture(), {"trace_depth": 7})
    assert np.array_equal(short.source_mass, long.source_mass[:4])
    assert np.array_equal(short.terminal_mass, long.terminal_mass[:4])


def test_zero_terminal_mass_leaves_normalized_depth_unavailable():
    probabilities = np.ones((1, 1, 1, 1), dtype=np.float64)
    fixture = LSPCFG(
        grammar="recursive",
        probabilities=probabilities,
        source_nodes={0: "A"},
        sink_nodes={0: ("A",)},
        source_to_sinks={0: (0,)},
        language_shape={"num_cores": 1, "max_num_states": 1},
    )
    analysis = analyze_language(fixture, {"trace_depth": 4})
    assert analysis.normalized_terminal_depth_mass is None
    assert analysis.reference_terminal_cdf is None
    assert analysis.reference_terminal_depth_mean is None
    assert analysis.reference_terminal_depth_variance is None
    assert analysis.residual_active_mass == 1.0


def test_entropy_split_and_recurrence_use_shared_trace():
    analysis = analyze_language(_fixture(), {"trace_depth": 4})
    assert np.allclose(
        analysis.hidden_latent_entropy,
        analysis.core_average_entropy - analysis.within_state_entropy,
    )
    assert analysis.recursive_components == ()
    assert analysis.recursive_component_mass.shape == (5, 0)
    assert analysis.summary()["trace_depth"] == 4


def test_recurrence_recovers_known_spectral_radius_components_and_mass():
    probabilities = np.zeros((1, 1, 3, 3), dtype=np.float64)
    probabilities[0, 0, 0, 0] = 1.0
    probabilities[0, 0, 1, 1] = 1.0
    probabilities[0, 0, 2, 2] = 1.0
    fixture = LSPCFG(
        grammar="semantic maps only; parser must not be consulted",
        probabilities=probabilities,
        source_nodes={0: "A", 1: "B", 2: "C"},
        sink_nodes={0: ("B",), 1: ("A",), 2: (2,)},
        source_to_sinks={0: (0,), 1: (1,), 2: (2,)},
        language_shape={"num_cores": 1, "max_num_states": 1},
    )
    analysis = analyze_language(fixture, {"trace_depth": 4})
    assert analysis.spectral_radius == 1.0
    assert analysis.recursive_components == ((0, 1),)
    assert np.array_equal(analysis.recursive_component_mass[:, 0], np.ones(5))
    repeated = analyze_language(fixture, {"trace_depth": 4})
    assert np.array_equal(analysis.source_mass, repeated.source_mass)
