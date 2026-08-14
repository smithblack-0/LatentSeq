# Using LatentSeq

LatentSeq is organized around three independently reusable objects: an `Agent` produces latent
state, a `Language` expresses that state as grammar terminals, and a `Vocabulary` realizes grammar
terminals as observable token IDs. `Generator` is only a convenience wrapper around the same three
objects.

## Construct reusable components

```python
from latentseq import Agent, Language, Vocabulary, validate_compatibility

agent_config = {
    "cores": [
        {"advancement_period": 0},
        {"advancement_period": {"function": "randint", "low": 6, "high": 21}},
    ],
    "defaults": {
        "hidden_size": 6,
        "num_connections": 3,
        "concentration": 1.0,
    },
}

language_config = {
    "grammar": {
        "terminal_pair_rules": 6,
        "parenthesis_rules": 4,
        "iteration_rules": 5,
        "branch_rules": 3,
        "max_terminals": 12,
        "max_nonterminals": 10,
        "language_shape": {
            "num_cores": 2,
            "max_num_states": 6,
        },
        "sampling_defaults": {
            "pairwise_odds": 6,
            "min_ppl": 1.2,
            "max_ppl": 8,
        },
    },
    "runtime": {
        "stack_depth": 128,
        "chunk_size": 64,
        "max_attempts": 5,
    },
}

vocabulary_config = {
    "vocabulary_size": 10_000,
    "elements_per_terminal": 1000,
    "pairwise_odds": 4,
}

agent = Agent.from_config(agent_config, device="cuda")
language = Language.from_config(language_config, device="cuda")
vocabulary = Vocabulary.from_config(language, vocabulary_config)
validate_compatibility(agent, language, vocabulary)
```

`Language.from_config` performs the complete CFG → LS-PCFG → LanguageCore construction internally.
Those representations remain available through the advanced API, but ordinary callers do not have
to carry them through their application.

## Generate aligned data later

Construction does not have to occur near sampling. Objects can be pooled, saved, loaded, selected,
and checked before they are used.

```python
latent = agent.sample(batch_size=512, length=2048)
grammar = language.sample(latent)
tokens = vocabulary.sample(latent, grammar)
```

The returned shapes are:

```text
latent   [batch, core, timestep]
grammar  [batch, timestep]
tokens   [batch, timestep]
```

This explicit three-stage form is useful when one representation must be retained or reused. The
one-call convenience form is:

```python
from latentseq import Generator

generator = Generator.from_config(
    {
        "agent": agent_config,
        "language": language_config,
        "vocabulary": vocabulary_config,
    },
    device="cuda",
)
sample = generator.sample(batch_size=512, length=2048)
```

`sample.latent`, `sample.grammar`, and `sample.tokens` are the same three representations.

## Pools and compatibility

Agents and Languages are intentionally compatible by structure rather than by shared identity. A
Language expects the same ordered core count, and every Agent core must emit state IDs that fit the
Language's latent-state width. A Vocabulary is sampled against one exact Language, so it also stores
that Language's semantic fingerprint.

This makes pool selection explicit and cheap to check:

```python
agent = agents[agent_index]
language = languages[language_index]
vocabulary = vocabularies[vocabulary_index]

validate_compatibility(agent, language, vocabulary)
```

A mismatch raises `CompatibilityError` with every detected incompatibility instead of silently
relying on coincidentally matching tensor shapes.

## Save and restore prepared objects

```python
agent.save_pretrained("agents/agent-0042")
language.save_pretrained("languages/language-0181")
vocabulary.save_pretrained("vocabularies/vocab-0181-a")

agent = Agent.from_pretrained("agents/agent-0042", device="cuda")
language = Language.from_pretrained(
    "languages/language-0181",
    device="cuda",
    stack_depth=256,
    chunk_size=128,
)
vocabulary = Vocabulary.from_pretrained(
    "vocabularies/vocab-0181-a",
    device="cuda",
)
```

`from_pretrained` restores the exact sampled reusable object. Language load-time overrides are
limited to `stack_depth`, `chunk_size`, and `max_attempts`, because those values change execution
resources rather than the sampled language identity. Construction controls such as
`pairwise_odds` cannot be overridden while loading.

Persistence does not represent a partially sampled trajectory. To continue latent state explicitly,
pass the final emitted states back to the Agent:

```python
next_latent = agent.sample(
    batch_size=latent.shape[0],
    length=2048,
    initial_markov_states=latent[:, :, -1],
)
```

Dwell phase is freshly sampled by that new call, as specified by the Agent contract.

## Randomness belongs to the caller

LatentSeq has no seed API and does not save, restore, rewind, or privately own NumPy or Torch RNG
state. New stochastic construction and sampling consume the ambient library streams. If an
experiment requires repeatability, seed or restore those libraries at the experiment boundary that
already owns randomness.

Object persistence therefore has no effect on the next NumPy or Torch random draw.

## Inspect a Language without rebuilding it

A `Language` retains its semantic sampled representation:

```python
analysis = language.analyze({"trace_depth": 128})
print(language.grammar)
```

For work that specifically needs construction IRs, import the advanced functions from
`latentseq.language`:

```python
from latentseq.language import (
    sample_cfg,
    sample_ls_pcfg,
    analyze_language,
    compile_language_core,
)
```

These functions expose the same boundaries used internally by `Language.from_config`; they are not
required for normal generation.
