# Testing guide

LatentSeq tests are organized by responsibility so failures can be traced back to one contract rather than treated as a single end-to-end signal.

## Commands

Install the development extras:

```bash
python -m pip install -e '.[dev]'
```

Run the full suite available on the current machine:

```bash
python -m pytest
```

Run the CPU-certifiable suite explicitly:

```bash
python -m pytest -m 'not cuda'
```

Run tests marked as compile-path certification:

```bash
python -m pytest -m compile
```

On a CUDA-capable environment, run the CUDA certification directly:

```bash
python -m pytest -m cuda
```

A CPU pass does not certify the CUDA contracts.

## Test ownership

| Test module | Primary responsibility |
|---|---|
| `tests/test_agent.py` | Core construction, dwell behavior, latent sampling, configuration merging, persistence, and Agent compile behavior. |
| `tests/test_cfg.py` | CFG schema/feasibility, Unold construction decisions, dense terminals, reachability/productivity, metadata, and reproducibility under externally controlled RNG. |
| `tests/test_language_sampling.py` | LS-PCFG tensor/maps, probability masks, sharpness controls, PPL/nats calibration, caps, core scaling, and product equivalence. |
| `tests/test_analysis.py` | Representative-model collapse order, offspring/emission operators, trace semantics, entropy decomposition, recurrence, and deterministic analysis. |
| `tests/test_compilation.py` | Shared runtime numbering, probability preservation, control nodes, terminal identity/executors, LIFO operands, and `LanguageCore` persistence. |
| `tests/test_language_runtime.py` | Stack, Reader, Recorder, CounterDecoder, Transitions, and InstructionDecoder unit behavior. |
| `tests/test_language_orchestration.py` | Language attempt construction, completion, chunk continuation, overflow/retry policy, and reusable Language persistence. |
| `tests/test_language_integration.py` | Real cooperating runtime components from LS to GS, repeated derivations, batch divergence, conditioning, and overflow behavior. |
| `tests/test_compilation_backend.py` | Full-graph compile execution, eager/compiled identity, custom sampling-op validation, and CUDA device-residency certification. |
| `tests/test_vocabulary.py` | Binding construction/coverage, lexical probability tables, conditioning, lookup/alignment, persistence, and compiled application. |
| `tests/test_persistence_and_compatibility.py` | Cross-component compatibility rules, fingerprints, pretrained round-trips, and runtime-only overrides. |
| `tests/test_public_api.py` | Top-level import surface and ordinary Agent -> Language -> Vocabulary / Generator usage. |
| `tests/test_rng_boundary.py` | External control of ambient RNG and the rule that persistence/load does not own RNG lifecycle. |

`tests/conftest.py` contains shared deterministic configurations and fixtures. It is test support, not an additional behavioral owner.

## Package verification

Repository CI performs two distinct checks:

1. **CPU behavior:** install the project with test dependencies and run `pytest -m 'not cuda'` plus source compilation.
2. **Distribution sanity:** build the source distribution and wheel, install the wheel into the runner, and perform a top-level import/API smoke check.

This split catches packaging failures that an editable source checkout can hide.

## CUDA status

CUDA tests are intentionally kept separate from the ordinary hosted CPU workflow. A CUDA test that is skipped because no GPU is present is an unexecuted certification, not a pass. The repository should continue to state CUDA status explicitly until an appropriate GPU runner or manual certification is available.
