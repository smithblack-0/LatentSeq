# Documentation map

LatentSeq separates documents that constrain development from documents that describe the repository as it exists today. This distinction is intentional: current code and tests must not be allowed to justify their own design after the fact.

## Authority and purpose

1. **Product specification** — the Paper 2 generator specification in `smithblack-0/ThesisProgram/2.md` defines the required generator behavior and architecture.
2. **Standards** — `docs/standards/` defines how LatentSeq code, tests, and reviews must be produced and evaluated. These rules are written independently of the current implementation.
3. **Guides** — `docs/guides/` describes the current repository: where responsibilities live, how to run the current test suite, and how to use the current public API.
4. **Changelog** — `CHANGELOG.md` records what changed over time. It is a historical inventory, not a source of design authority.

If a guide reveals that the current implementation violates a standard or the product specification, the implementation or guide must be corrected. The standard is not rewritten merely to justify the implementation.

## Standards

- `standards/coding.md` — code quality, architecture, documentation, configuration, and refactoring expectations.
- `standards/testing.md` — TDD, behavioral-test design, integration/backend certification, and evidence requirements.
- `standards/review.md` — reviewability, changelists, evidence maps, risk/gap disclosure, and approval criteria.

## Guides

- `guides/system.md` — current architecture, package layout, responsibility boundaries, and public API structure.
- `guides/testing.md` — current test locations, commands, markers, and CI jobs.
- `guides/usage.md` — current user-facing construction, persistence, pooling, and sampling examples.

For contribution setup and the shortest path into these documents, see `CONTRIBUTING.md`.
