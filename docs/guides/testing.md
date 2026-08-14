# Testing guide

This guide describes the current LatentSeq test suite and CI layout. The rules that tests must satisfy are in `../standards/testing.md`.

## Commands

Install development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Run the suite available on the current machine:

```bash
python -m pytest
```

Run the CPU-certifiable suite explicitly:

```bash
python -m pytest -m 'not cuda'
```

Run compile-path tests:

```bash
python -m pytest -m compile
```

On a CUDA-capable environment, run CUDA certification directly:

```bash
python -m pytest -m cuda
```

A CUDA test skipped because no GPU is present is unexecuted certification, not a pass.

## Current test map

| Test module | Current responsibility |
|---|---|
| `tests/test_agent.py` | Core construction, dwell behavior, latent sampling, configuration merging, persistence, and Agent compile behavior. |
| `tests/test_cfg.py` | CFG schema/feasibility, construction decisions, terminal identity, reachability/productivity, metadata, and externally seeded reproducibility. |
| `tests/test_language_sampling.py` | LS-PCFG tensors/maps, masks, sharpness controls, PPL/nats calibration, caps, core scaling, and probability-product behavior. |
| `tests/test_analysis.py` | Representative collapse, offspring/emission operators, trace semantics, entropy decomposition, recurrence, and deterministic analysis. |
| `tests/test_compilation.py` | Runtime numbering, probability preservation, control nodes, terminal executors/identity, LIFO operands, and LanguageCore persistence. |
| `tests/test_language_runtime.py` | Stack, Reader, Recorder, CounterDecoder, Transitions, and InstructionDecoder behavior. |
| `tests/test_language_orchestration.py` | Language attempts, completion, chunk continuation, overflow/retry, and reusable Language persistence. |
| `tests/test_language_integration.py` | Real cooperating runtime components from LS to GS, repeated derivations, batch divergence, conditioning, and overflow. |
| `tests/test_compilation_backend.py` | Full-graph compile execution, eager/compiled behavior, custom sampling-op validation, and CUDA device-residency certification. |
| `tests/test_vocabulary.py` | Bindings/coverage, lexical probability tables, conditioning, alignment, persistence, and compiled application. |
| `tests/test_persistence_and_compatibility.py` | Current cross-component compatibility behavior, pretrained round-trips, and runtime-only overrides. |
| `tests/test_public_api.py` | Top-level import surface and ordinary component/Generator usage. |
| `tests/test_rng_boundary.py` | External RNG control and persistence/load non-ownership of RNG lifecycle. |

`tests/conftest.py` contains shared test configurations and fixtures.

This table is descriptive. If tests move during a refactor, update the guide; do not preserve a poor test boundary merely to keep this table stable.

## Current CI

`.github/workflows/ci.yml` runs two jobs on pull requests and pushes to `main`:

1. **cpu-tests** — installs test dependencies, compiles source, and runs `pytest -m 'not cuda'`.
2. **package** — builds source/wheel distributions, installs the wheel, then smoke-tests the installed top-level API outside the source checkout.

CUDA certification is intentionally not claimed by this hosted CPU workflow.
