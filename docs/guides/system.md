# System guide

This guide describes the current LatentSeq repository and is intended for navigation. It does not define engineering standards and does not override the Paper 2 generator specification.

## Data path

LatentSeq keeps latent process, grammar realization, and vocabulary realization separate:

```text
Agent -> Latent Sample (LS) -> Language -> Grammar Sample (GS) -> Vocabulary -> Token Sample (TS)
```

Vocabulary consumes both LS and GS so lexical realization remains conditioned on the same latent states that produced the grammar sequence.

## Language construction path

A Language is built through semantic construction stages before execution:

```text
CFG configuration
    -> CFG sampling
    -> CFG
    -> LS-PCFG sampling
    -> LS-PCFG
    -> compilation
    -> LanguageCore
```

LS-PCFG analysis is an optional deterministic branch used for inspection rather than an input to compilation.

Normal users do not need to invoke this pipeline manually. `Language.from_config(...)` orchestrates it while the semantic IR stages remain available under `latentseq.language` for research and inspection.

## Component responsibilities

### Agent

`src/latentseq/agent.py`

Constructs and executes the latent HMM cores that produce LS. The public `Agent` composes ordered cores and exposes construction, sampling, and pretrained persistence.

### Language construction and analysis

`src/latentseq/language/`

- `cfg.py` — CFG construction and parsing.
- `sampling.py` — CFG -> latent-conditioned LS-PCFG probability tables.
- `ir.py` — semantic IR containers.
- `analysis.py` — deterministic analysis of the LS-PCFG representation.
- `compilation.py` — LS-PCFG -> `LanguageCore` lowering.

### Language runtime

- `runtime.py` — tensor pushdown runtime primitives.
- `api.py` — public Language construction/runtime facade and persistence lifecycle.

The runtime executes a compiled LanguageCore against LS to produce GS.

### Vocabulary

`src/latentseq/vocabulary.py`

Constructs terminal-to-vocabulary bindings and latent-conditioned lexical distributions, then lowers aligned LS + GS to TS.

### Compatibility and orchestration

- `compatibility.py` — explicit cross-component compatibility checks.
- `generator.py` — optional whole-pipeline convenience orchestration.
- `persistence.py` — shared local persistence helpers.

`Generator` coordinates Agent, Language, and Vocabulary; it does not replace their independent APIs.

## Public API

The normal top-level surface is:

```python
from latentseq import (
    Agent,
    Language,
    Vocabulary,
    Generator,
    validate_compatibility,
)
```

The component-oriented path remains first-class:

```python
agent = Agent.from_config(agent_config, device="cuda")
language = Language.from_config(language_config, device="cuda")
vocabulary = Vocabulary.from_config(language, vocabulary_config)

validate_compatibility(agent, language, vocabulary)

latent = agent.sample(batch_size=512, length=2048)
grammar = language.sample(latent)
tokens = vocabulary.sample(latent, grammar)
```

See `usage.md` for persistence, pooling, and advanced examples.

## Persistence boundary

`save_pretrained(...)` / `from_pretrained(...)` preserve reusable sampled components. They do not represent an in-progress sequence trajectory.

Language loading may accept runtime-resource overrides where supported without reconstructing the saved stochastic language.

Randomness is ambient NumPy/Torch behavior: stochastic construction/sampling consumes those streams, while persistence does not store or restore RNG lifecycle state.

## Repository map

```text
src/latentseq/              package implementation
tests/                      behavior/integration/backend tests
docs/standards/             development constraints
docs/guides/                current repository/user guides
.github/workflows/ci.yml     hosted CPU/package checks
CHANGELOG.md                 durable change history
CONTRIBUTING.md              contributor entry point
pyproject.toml               package/test metadata
```
