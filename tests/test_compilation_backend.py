"""Backend certification tests require graph-pure compiled sampling and eager/compiled identity."""

import numpy as np
import pytest
import torch

from latentseq import Agent, Language, Vocabulary
from latentseq.agent import build_core
from latentseq.language.ir import LSPCFG
from latentseq.language.runtime import build_runtime_attempt, decode_chunk


def _stochastic_language(device: torch.device) -> Language:
    probabilities = np.zeros((1, 1, 1, 2), dtype=np.float64)
    probabilities[0, 0, 0] = [0.5, 0.5]
    return Language.from_ls_pcfg(
        LSPCFG(
            grammar="stochastic",
            probabilities=probabilities,
            source_nodes={0: "A"},
            sink_nodes={0: (0,), 1: (1,)},
            source_to_sinks={0: (0, 1)},
            language_shape={"num_cores": 1, "max_num_states": 1},
        ),
        {"stack_depth": 8, "chunk_size": 16, "max_attempts": 2},
        device,
    )


@pytest.mark.compile
def test_decode_chunk_compiles_fullgraph_and_matches_eager_state():
    language = _stochastic_language(torch.device("cpu"))
    latent = torch.zeros((2, 1, 4), dtype=torch.int64)
    compiled = torch.compile(decode_chunk, dynamic=False, fullgraph=True)

    # Warm compilation outside the compared random streams.
    warm = build_runtime_attempt(latent, language.core, 8)
    torch.manual_seed(999)
    compiled(warm, language.transitions, language.instruction_decoder, 16)

    torch.manual_seed(123)
    eager = build_runtime_attempt(latent, language.core, 8)
    decode_chunk(eager, language.transitions, language.instruction_decoder, 16)

    torch.manual_seed(123)
    optimized = build_runtime_attempt(latent, language.core, 8)
    compiled(optimized, language.transitions, language.instruction_decoder, 16)

    assert torch.equal(eager.recorder.retrieve(), optimized.recorder.retrieve())
    assert torch.equal(eager.stack.stack, optimized.stack.stack)
    assert torch.equal(eager.stack.stack_position, optimized.stack.stack_position)
    assert torch.equal(eager.stack.overflow, optimized.stack.overflow)
    assert torch.equal(eager.counter.latent_position, optimized.counter.latent_position)


@pytest.mark.compile
def test_agent_sampling_with_random_dwell_compiles_fullgraph_and_matches_eager():
    core = build_core(
        {
            "hidden_size": 4,
            "num_connections": 2,
            "advancement_period": {"function": "randint", "low": 0, "high": 3},
            "concentration": 1.0,
        },
        torch.device("cpu"),
    )
    compiled = torch.compile(core.draw_samples, dynamic=False, fullgraph=True)
    torch.manual_seed(77)
    compiled(2, 12)  # warm
    torch.manual_seed(8)
    eager = core.draw_samples(2, 12)
    torch.manual_seed(8)
    optimized = compiled(2, 12)
    assert torch.equal(eager, optimized)


@pytest.mark.compile
def test_vocabulary_application_compiles_fullgraph_and_matches_eager(
    language_config, vocabulary_config, device
):
    np.random.seed(4)
    torch.manual_seed(4)
    language = Language.from_config(language_config, device=device)
    vocabulary = Vocabulary.from_config(language, vocabulary_config)
    latent = torch.zeros((2, language.num_cores, 8), dtype=torch.int64)
    grammar = torch.arange(8).remainder(language.grammar_terminal_count).repeat(2, 1)
    compiled = torch.compile(vocabulary._sample_unchecked, dynamic=False, fullgraph=True)
    torch.manual_seed(99)
    compiled(latent, grammar)  # warm
    torch.manual_seed(7)
    eager = vocabulary._sample_unchecked(latent, grammar)
    torch.manual_seed(7)
    optimized = compiled(latent, grammar)
    assert torch.equal(eager, optimized)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend is not available")
def test_cuda_public_pipeline_stays_device_resident(agent_config, language_config, vocabulary_config):
    device = torch.device("cuda")
    np.random.seed(12)
    torch.manual_seed(12)
    agent = Agent.from_config(agent_config, device=device)
    language = Language.from_config(language_config, device=device)
    vocabulary = Vocabulary.from_config(language, vocabulary_config)
    latent = agent.sample(2, 16)
    grammar = language.sample(latent)
    tokens = vocabulary.sample(latent, grammar)
    assert latent.device.type == grammar.device.type == tokens.device.type == "cuda"
    assert all(core.transition_table.device.type == "cuda" for core in agent.cores)
    assert language.core.transition_tables.device.type == "cuda"
    assert vocabulary.probabilities.device.type == "cuda"


@pytest.mark.compile
def test_ambient_sampling_custom_ops_pass_torch_opcheck():
    from latentseq._sampling_ops import _ambient_multinomial, _ambient_randint

    anchor = torch.empty(0)
    probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    randint_result = torch.library.opcheck(_ambient_randint, (anchor, 0, 3, 4))
    multinomial_result = torch.library.opcheck(_ambient_multinomial, (probabilities,))
    assert set(randint_result.values()) == {"SUCCESS"}
    assert set(multinomial_result.values()) == {"SUCCESS"}


@pytest.mark.compile
def test_complete_language_generation_matches_eager_and_compiled_chunks():
    eager_language = _stochastic_language(torch.device("cpu"))
    compiled_language = _stochastic_language(torch.device("cpu"))
    compiled_language.decode_chunk_function = torch.compile(
        decode_chunk, dynamic=False, fullgraph=True
    )
    latent = torch.zeros((2, 1, 6), dtype=torch.int64)

    torch.manual_seed(999)
    compiled_language.sample(latent)  # warm compile outside compared streams

    torch.manual_seed(121)
    eager = eager_language.sample(latent)
    torch.manual_seed(121)
    optimized = compiled_language.sample(latent)
    assert torch.equal(eager, optimized)
