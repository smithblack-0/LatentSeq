# LatentSeq

LatentSeq generates aligned latent-state, grammar-state, and observable-token sequences from independently reusable stochastic components.

For complete configuration, pooling, persistence, and advanced-API examples, see `docs/usage.md`. The current implementation is the initial build and is still under review; backend verification status is tracked in `CHANGELOG.md` and `docs/testing.md`.

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

## Repository documentation

- `CHANGELOG.md` — subsystem-by-subsystem change inventory and verification status.
- `docs/review-guide.md` — implementation-to-test map for auditing the initial build.
- `docs/testing.md` — test ownership, commands, package checks, and CUDA certification boundary.
- `docs/development.md` — source layout, responsibility boundaries, and development workflow.
- `CONTRIBUTING.md` — concise entry point for changes to the repository.

Hosted CI certifies the CPU/package paths only. CUDA certification is kept explicit and separate rather than inferred from CPU success.
