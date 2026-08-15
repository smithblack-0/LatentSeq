"""Agent tests certify sparse construction, dwell semantics, resumption, persistence, and config merging."""

import torch
import pytest

from latentseq import Agent
from latentseq.agent import Core, build_core


def _reachable(adjacency: torch.Tensor, start: int) -> set[int]:
    seen = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for nxt in torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist():
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def test_core_construction_sparse_strongly_connected_and_reproducible(device):
    config = {
        "hidden_size": 6,
        "num_connections": 3,
        "advancement_period": 0,
        "concentration": 1.0,
    }
    torch.manual_seed(41)
    first = build_core(config, device)
    torch.manual_seed(41)
    second = build_core(config, device)
    table = first.transition_table
    assert table.shape == (6, 6)
    assert table.dtype == torch.float32
    assert torch.allclose(table.sum(dim=1), torch.ones(6))
    assert torch.equal((table > 0).sum(dim=1), torch.full((6,), 3))
    for start in range(6):
        assert _reachable(table > 0, start) == set(range(6))
    assert torch.equal(first.transition_table, second.transition_table)


def test_agent_shape_and_resumption(agent_config, device):
    agent = Agent.from_config(agent_config, device=device)
    initial = torch.tensor([[1, 3], [2, 0]], dtype=torch.int64, device=device)
    latent = agent.sample(2, 5, initial_markov_states=initial)
    assert latent.shape == (2, 2, 5)
    assert latent.dtype == torch.int64
    assert torch.equal(latent[:, :, 0], initial)


def test_constant_dwell_semantics(device):
    transition = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
    core = Core(
        num_states=2,
        transition_table=transition,
        advancement_period=2,
        dwell_sampler=lambda width: torch.full((width,), 2, dtype=torch.int64, device=device),
        construction_config={
            "hidden_size": 2,
            "num_connections": 2,
            "advancement_period": 2,
            "concentration": 1.0,
        },
        device=device,
    )
    sample = core.draw_samples(1, 10, torch.tensor([0], device=device))
    assert sample.tolist()[0] == [0, 0, 0, 1, 1, 1, 0, 0, 0, 1]


def test_transition_changes_follow_only_nonzero_edges(agent_config, device):
    torch.manual_seed(9)
    agent = Agent.from_config(agent_config, device=device)
    latent = agent.sample(8, 60)
    for core_index, core in enumerate(agent.cores):
        states = latent[:, core_index]
        changed = states[:, 1:] != states[:, :-1]
        if changed.any():
            before = states[:, :-1][changed]
            after = states[:, 1:][changed]
            assert torch.all(core.transition_table[before, after] > 0)


def test_low_concentration_has_lower_mean_row_entropy(device):
    base = {
        "hidden_size": 512,
        "num_connections": 8,
        "advancement_period": 0,
    }
    torch.manual_seed(80)
    low = build_core({**base, "concentration": 0.1}, device)
    torch.manual_seed(80)
    high = build_core({**base, "concentration": 10.0}, device)

    def entropy(table):
        positive = table > 0
        safe = torch.where(positive, table, torch.ones_like(table))
        return -(torch.where(positive, table * torch.log(safe), torch.zeros_like(table))).sum(1).mean()

    assert entropy(low.transition_table) < entropy(high.transition_table)


def test_configuration_defaults_are_explicit_merge_only(agent_config, device):
    full = {
        "cores": [
            {**agent_config["defaults"], **agent_config["cores"][0]},
            {**agent_config["defaults"], **agent_config["cores"][1]},
        ],
        "defaults": {},
    }
    torch.manual_seed(5)
    merged = Agent.from_config(agent_config, device=device)
    torch.manual_seed(5)
    explicit = Agent.from_config(full, device=device)
    for left, right in zip(merged.cores, explicit.cores, strict=True):
        assert left.construction_config == right.construction_config
        assert torch.equal(left.transition_table, right.transition_table)

    broken = {"cores": agent_config["cores"], "defaults": {"hidden_size": 4}}
    with pytest.raises(ValueError, match="missing"):
        Agent.from_config(broken, device=device)


def test_malformed_dwell_sampler_fails_during_new_construction(device):
    config = {
        "hidden_size": 4,
        "num_connections": 2,
        "advancement_period": {"function": "rand", "dtype": torch.float32},
        "concentration": 1.0,
    }
    with pytest.raises(ValueError, match="integer"):
        build_core(config, device)


def test_bad_initial_states_raise(agent_config, device):
    agent = Agent.from_config(agent_config, device=device)
    with pytest.raises(ValueError):
        agent.sample(2, 5, torch.zeros((2, 3), dtype=torch.int64))
    with pytest.raises(ValueError):
        agent.sample(2, 5, torch.tensor([[9, 0], [0, 0]], dtype=torch.int64))


def test_agent_configuration_is_exact_keyed(agent_config, device):
    with pytest.raises(ValueError):
        Agent.from_config({**agent_config, "unknown": 1}, device=device)


def test_agent_shape_edge_cases_and_explicit_core_override(device):
    config = {
        "cores": [{"advancement_period": 0, "hidden_size": 3}],
        "defaults": {
            "hidden_size": 4,
            "num_connections": 2,
            "concentration": 1.0,
        },
    }
    agent = Agent.from_config(config, device=device)
    latent = agent.sample(batch_size=1, length=1)
    assert latent.shape == (1, 1, 1)
    assert latent.dtype == torch.int64
    assert latent.device == device
    assert agent.state_signature == (3,)


def test_dwell_validation_rejects_unknown_wrong_shape_and_negative(device):
    base = {
        "hidden_size": 4,
        "num_connections": 2,
        "concentration": 1.0,
    }
    with pytest.raises(ValueError, match="unknown"):
        build_core({**base, "advancement_period": {"function": "not_a_torch_function"}}, device)
    with pytest.raises(ValueError, match="batch-shaped"):
        build_core(
            {
                **base,
                "advancement_period": {
                    "function": "zeros",
                    "dtype": torch.int64,
                    "size": (2, 2),
                },
            },
            device,
        )
    with pytest.raises(ValueError, match="negative"):
        build_core(
            {
                **base,
                "advancement_period": {
                    "function": "full",
                    "fill_value": -1,
                    "dtype": torch.int64,
                },
            },
            device,
        )
