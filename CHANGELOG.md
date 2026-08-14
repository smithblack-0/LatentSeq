# Changelog

This file records user-visible and maintainer-relevant changes to LatentSeq. The project has not made a packaged release yet; the implementation below is the initial build currently under review.

## Unreleased

### Repository and package

- Added the installable `latentseq` package using a `src/` layout and `pyproject.toml` metadata.
- Added a narrow top-level API exposing `Agent`, `Language`, `Vocabulary`, `Generator`, and `validate_compatibility`.
- Added Hugging Face-style `from_config`, `save_pretrained`, and `from_pretrained` lifecycle methods for reusable sampled components.
- Added usage, development, testing, and review documentation so the implementation can be audited by subsystem rather than as one monolithic change.
- Added CPU/package CI that builds the distribution, installs the wheel, and runs the non-CUDA test suite. CUDA remains a separate explicit certification target.

### Agent

- Added sparse strongly connected HMM-core construction with configurable out-degree and Dirichlet transition weights.
- Added constant and configured Torch dwell processes.
- Added batch-parallel latent-state sampling and ordered multi-core composition.
- Added sampled-structure persistence that restores transition tables and dwell configuration without reconstructing them stochastically.

Primary implementation: `src/latentseq/agent.py`  
Primary tests: `tests/test_agent.py`

### CFG construction

- Added exact-keyed CFG configuration validation and feasibility checks.
- Added Unold-style grammar construction across terminal-pair, parenthesis, iteration, and branch productions.
- Added dense terminal identity, productive/reachable grammar guarantees, hanging-source handling, duplicate local redraw, and metadata rendering.
- Added CFG parsing used by later semantic stages.

Primary implementation: `src/latentseq/language/cfg.py`  
Primary tests: `tests/test_cfg.py`

### Language sampling

- Added CFG-to-LS-PCFG sampling with latent-conditioned raw probability tables.
- Added `pairwise_odds`, `ppl`, and `nats` controls, inherited perplexity caps, numerical entropy calibration, and per-core scaling.
- Added semantic source/sink maps and normalized masked probability rows.

Primary implementation: `src/latentseq/language/sampling.py`  
Primary tests: `tests/test_language_sampling.py`

### Language analysis

- Added the state-average-then-intersect representative model required by the specification.
- Added expected nonterminal-offspring and terminal-emission operators and shared finite-depth propagation.
- Added terminal-depth, entropy, hidden-latent-entropy, spectral-radius, and recursive-component summaries.

Primary implementation: `src/latentseq/language/analysis.py`  
Primary tests: `tests/test_analysis.py`

### Language compilation

- Added deterministic LS-PCFG to `LanguageCore` lowering with a shared runtime numbering for transition and instruction tables.
- Added start/done control nodes, reusable compound-terminal executors, semantic terminal identity preservation, and LIFO operand lowering.
- Added exact compiled-core persistence.

Primary implementation: `src/latentseq/language/compilation.py`  
Primary tests: `tests/test_compilation.py`

### Language runtime

- Added fixed-shape `Stack`, `Reader`, `Recorder`, `CounterDecoder`, `Transitions`, and `InstructionDecoder` runtime components.
- Added batch-parallel pushdown decoding, start-sentinel repeated derivations, done-lane inertness, and fixed-memory overflow detection.
- Added fixed-trip compiled decode chunks with host-side completion/overflow decisions and whole-attempt retry semantics.
- Added `Language` orchestration, runtime configuration, pretrained restoration, and runtime-only load overrides.

Primary implementation: `src/latentseq/language/runtime.py`, `src/latentseq/language/api.py`  
Primary tests: `tests/test_language_runtime.py`, `tests/test_language_orchestration.py`, `tests/test_language_integration.py`, `tests/test_compilation_backend.py`

### Vocabulary

- Added exact-width per-terminal vocabulary bindings with global vocabulary coverage.
- Added latent-conditioned lexical probability tables using the same core-scaled sharpness semantics as Language sampling.
- Added batch/position-parallel LS + GS to TS lowering through latent-factor conditioning.
- Added exact vocabulary persistence and restoration.

Primary implementation: `src/latentseq/vocabulary.py`  
Primary tests: `tests/test_vocabulary.py`

### Compatibility, orchestration, and persistence

- Added structural Agent/Language compatibility validation and Language/Vocabulary identity validation.
- Added stable Language fingerprints so independently stored vocabularies cannot silently attach to a different but shape-compatible Language.
- Added `Generator` as optional orchestration over the independently usable Agent, Language, and Vocabulary components.
- Kept random-number-generator lifecycle external to LatentSeq; stochastic calls consume ambient NumPy/Torch streams while persistence stores no RNG state.

Primary implementation: `src/latentseq/compatibility.py`, `src/latentseq/generator.py`, `src/latentseq/persistence.py`  
Primary tests: `tests/test_persistence_and_compatibility.py`, `tests/test_public_api.py`, `tests/test_rng_boundary.py`

### Verification status

- CPU behavior and compile-path tests pass locally.
- Source compilation and editable installation pass locally.
- Distribution build and installed-wheel smoke checking are part of repository CI.
- CUDA device-residency/certification remains intentionally unverified until run on a CUDA backend; CPU CI is not a substitute for that requirement.
