# Development guide

This document is the maintainer map for LatentSeq. It describes where responsibilities live and how to make changes without reconstructing the architecture from the source tree.

## Repository layout

```text
src/latentseq/              installable package
  agent.py                  latent HMM construction and sampling
  compatibility.py          cross-component compatibility checks
  generator.py              optional end-to-end orchestration
  persistence.py            shared persistence helpers
  vocabulary.py             lexical construction and runtime lowering
  language/
    cfg.py                   CFG construction and parsing
    sampling.py              CFG -> LS-PCFG probability sampling
    analysis.py              deterministic LS-PCFG analysis
    compilation.py           LS-PCFG -> LanguageCore lowering
    runtime.py               pushdown runtime primitives
    api.py                   public Language construction/runtime facade
    ir.py                    semantic construction IR containers

tests/                      behavior and backend certification

docs/usage.md               user-facing examples
docs/testing.md             test map and commands
docs/review-guide.md        subsystem-by-subsystem review map
CHANGELOG.md                 durable change inventory
```

## Responsibility boundaries

The package deliberately separates reusable stochastic components from the orchestration that creates or combines them.

- `Agent` owns only latent-state generation.
- Language construction owns CFG sampling, LS-PCFG sampling, optional analysis, and compilation into `LanguageCore`.
- `Language` owns execution of a compiled language against a Latent Sample.
- `Vocabulary` owns the sampled lexical mapping and LS + GS -> TS lowering.
- `Generator` is convenience orchestration; it does not replace the independent component APIs.
- Compatibility checks are explicit and live at the public boundary rather than being hidden inside unrelated sampling calls.

Construction IRs remain inspectable so research workflows can stop between stages. Normal users do not need to invoke those stages manually because `Language.from_config(...)` orchestrates them.

## Configuration and failure policy

Configurations are explicit contracts. Required values are required; the package does not use hidden fallback defaults for construction/runtime fields covered by the specification. Invalid configuration should fail near the construction boundary rather than being silently repaired.

Runtime-policy overrides on `Language.from_pretrained(...)` may change execution resources such as stack depth or chunk size. They do not resample or modify the saved semantic Language identity.

## Randomness boundary

LatentSeq consumes the ambient NumPy/Torch random streams when stochastic construction or sampling occurs. RNG lifecycle is not package state: LatentSeq does not provide a seed API and does not persist, restore, rewind, or associate RNG state with saved components.

The compile-support sampling wrappers in `_sampling_ops.py` are compiler boundaries around ordinary Torch sampling calls; they do not create a package-owned generator or RNG stream.

## Development setup

Create an environment with Python 3.11 or later, then install development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Run the ordinary CPU suite:

```bash
python -m pytest -m 'not cuda'
```

Run source compilation:

```bash
python -m compileall -q src
```

Build distributions:

```bash
python -m build
```

See `docs/testing.md` for backend-specific certification and the ownership of each test module.

## Change discipline

When changing a subsystem:

1. Identify its public/IR/runtime contract in the specification and the corresponding row in `docs/review-guide.md`.
2. Change or add behavior tests at the boundary that owns the contract.
3. Implement the behavior without moving responsibilities across module boundaries accidentally.
4. Run the narrow tests first, then the non-CUDA suite.
5. Update `CHANGELOG.md` when the externally visible or maintainer-relevant behavior changes.
6. If compile or CUDA behavior is affected, run the matching backend certification rather than treating ordinary CPU tests as a substitute.

No release or publishing automation is part of the initial repository build.
