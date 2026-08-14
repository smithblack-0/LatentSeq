# LatentSeq

LatentSeq generates aligned latent-state, grammar-state, and observable-token sequences from independently reusable stochastic components.

For complete configuration, pooling, persistence, and advanced-API examples, see `docs/usage.md`.

## Normal use

```python
from latentseq import Agent, Language, Vocabulary, validate_compatibility

agent = Agent.from_config(agent_config, device="cuda")
language = Language.from_config(language_config, device="cuda")
vocabulary = Vocabulary.from_config(language, vocabulary_config)

validate_compatibility(agent, language, vocabulary)

latent = agent.sample(batch_size=512, length=2048)
grammar = language.sample(latent)
tokens = vocabulary.sample(latent, grammar)
```

The three components can be constructed, saved, loaded, pooled, and recombined independently. LatentSeq does not own or persist RNG state; stochastic calls consume the ambient NumPy and Torch RNG streams.

## Persistence

```python
agent.save_pretrained("agent-0042")
language.save_pretrained("language-0181")
vocabulary.save_pretrained("vocab-0181-a")

agent = Agent.from_pretrained("agent-0042", device="cuda")
language = Language.from_pretrained(
    "language-0181",
    device="cuda",
    stack_depth=256,
    chunk_size=128,
)
vocabulary = Vocabulary.from_pretrained("vocab-0181-a", device="cuda")
```

`Language.from_pretrained` accepts only runtime-policy overrides. Construction-time identity is restored exactly.

## Whole-pipeline convenience

```python
from latentseq import Generator

generator = Generator.from_config(config, device="cuda")
sample = generator.sample(batch_size=32, length=4096)

sample.latent
sample.grammar
sample.tokens
```

`Generator` is orchestration over the same three independently usable components.

## Advanced construction

The semantic construction pipeline remains available under `latentseq.language` for research and inspection:

```python
from latentseq.language import (
    sample_cfg,
    sample_ls_pcfg,
    analyze_language,
    compile_language_core,
)
```

These lower-level functions are not required for normal use.
