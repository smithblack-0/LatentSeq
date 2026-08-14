# Initial build review guide

The initial LatentSeq build is large enough that it should be reviewed as a set of subsystem changes rather than as one undifferentiated diff. This guide maps each change area to its implementation and its primary evidence.

## Review order

The suggested order follows the data path and keeps lower-level contracts visible before their orchestration layers:

1. Agent
2. CFG construction
3. Language sampling
4. Language analysis
5. Language compilation
6. Language runtime
7. Vocabulary
8. Compatibility, persistence, and public orchestration
9. Packaging and repository infrastructure

## Change map

| Area | What changed | Implementation | Primary tests |
|---|---|---|---|
| Agent | Sparse strongly connected latent cores, dwell policies, batch sampling, ordered composition, persistence. | `src/latentseq/agent.py` | `tests/test_agent.py` |
| CFG construction | Exact config/feasibility validation, Unold-style rule construction, symbol identity, reachability/productivity, parser/metadata. | `src/latentseq/language/cfg.py` | `tests/test_cfg.py` |
| Language sampling | CFG -> LS-PCFG probability tables, source/sink maps, pairwise-odds/PPL/nats controls, core scaling. | `src/latentseq/language/sampling.py`, `src/latentseq/language/ir.py` | `tests/test_language_sampling.py` |
| Language analysis | Representative collapse, offspring/emission matrices, depth trace, entropy and recurrence diagnostics. | `src/latentseq/language/analysis.py` | `tests/test_analysis.py` |
| Language compilation | Shared semantic/runtime numbering, transition/instruction lowering, start/done nodes, terminal executors, compiled persistence. | `src/latentseq/language/compilation.py` | `tests/test_compilation.py` |
| Language runtime | Fixed-shape pushdown helpers, conditioning, emitted-token counter, chunked execution, completion/overflow/retry. | `src/latentseq/language/runtime.py`, `src/latentseq/language/api.py` | `tests/test_language_runtime.py`, `tests/test_language_orchestration.py`, `tests/test_language_integration.py`, `tests/test_compilation_backend.py` |
| Vocabulary | Terminal bindings, coverage repair, latent-conditioned lexical tables, LS + GS -> TS sampling, persistence. | `src/latentseq/vocabulary.py` | `tests/test_vocabulary.py` |
| Compatibility | Agent/Language shape rules and Language/Vocabulary identity/fingerprint checks. | `src/latentseq/compatibility.py` | `tests/test_persistence_and_compatibility.py` |
| Persistence | Shared pretrained-file helpers; exact reusable-object restoration without stochastic reconstruction. | `src/latentseq/persistence.py`, component save/load methods | `tests/test_persistence_and_compatibility.py`, `tests/test_rng_boundary.py` |
| Public API | Narrow top-level imports, factories, pretrained lifecycle, `Generator`, normal three-stage sampling flow. | `src/latentseq/__init__.py`, `src/latentseq/generator.py`, component API classes | `tests/test_public_api.py` |
| Package/repo | Installable `src` package, usage/development/testing docs, changelog, CI, ignore rules, wheel build check. | `pyproject.toml`, `README.md`, `.github/workflows/ci.yml`, `docs/`, `CHANGELOG.md`, `.gitignore` | CI package job plus existing suite |

## Cross-cutting contracts to inspect

### Compatibility

A Language is intentionally reusable across compatible Agents, so Agent/Language compatibility is structural rather than identity-based. Vocabulary is sampled against a specific Language and records a Language fingerprint to prevent accidental attachment to another shape-compatible Language.

### Persistence

`save_pretrained` / `from_pretrained` restore reusable sampled machinery. They do not restore a partially generated trajectory and do not own RNG lifecycle. Language permits explicit runtime-resource overrides at load time without changing its sampled semantic identity.

### Compile behavior

The runtime is designed around fixed-shape compiled chunks. Backend-specific tests live in `tests/test_compilation_backend.py`. CUDA certification remains a separate explicit requirement and is not inferred from CPU execution.

## Known verification gap

The current development environment has no CUDA device. The CUDA-residency/certification test therefore remains unexecuted locally. This is the known backend verification gap for the initial build; it should not be silently converted into a CPU pass.
