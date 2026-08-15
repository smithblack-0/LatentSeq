# Testing standards

Testing in LatentSeq is part of the development process, not retrospective coverage added after implementation. These standards define what counts as useful evidence independently of the current test files.

## TDD development loop

For a non-trivial unit of work, development proceeds from contract to evidence to implementation:

1. State what is being built or changed.
2. Determine what it means for that change to be correct, maintainable, concise, fast, and effective.
3. Define the tests/evidence required to establish those contracts before implementation drives the test design.
4. Implement the behavior and change surrounding structure when the correct boundary requires it.
5. Add missing evidence discovered during implementation.
6. Reevaluate the design after tests pass; if the quality balance is poor, return to the contract/design step rather than patching around it.

Complex work is expected to require iteration.

## Tests specify behavior, not implementation history

- Tests should assert externally meaningful behavior, representation contracts, invariants, and failure semantics.
- Do not encode private call order, incidental helper structure, or exact internal decomposition unless that behavior is itself a required contract.
- A test written only because the implementation happens to work that way is suspect.
- When a design changes legitimately, refactor obsolete tests rather than preserving them as accidental architecture.
- Test names and fixtures should communicate the contract being exercised, not merely the function being called.

Tests must be capable of rejecting a bad design. They must not be derived so mechanically from finished code that they only prove the implementation is self-consistent.

## Evidence at the owning boundary

- Put the primary behavior test at the boundary that owns the behavior.
- Unit tests should isolate a meaningful contract, not fragment the system into tests of trivial statements.
- Integration tests should exercise real cooperating components where cross-boundary behavior matters.
- Prefer real tensors and real component implementations over mocks when practical.
- Use dependency injection/factories to isolate genuinely external or replaceable behavior instead of monkey-patching internals.
- End-to-end tests are valuable for composition, but they do not replace focused evidence for the contracts that make the composition trustworthy.

## Randomness

- Tests may seed NumPy/Torch externally to obtain reproducible evidence.
- Tests must not require LatentSeq to own, persist, restore, or rewind RNG state unless the product specification is deliberately changed to make that a package responsibility.
- Randomized algorithms need structural/statistical tests appropriate to their contracts; one fixed seed is not sufficient evidence for properties such as reachability, coverage, normalization, or feasibility.
- Retry paths must be tested for their specified interaction with random draws rather than assuming a retry rewinds randomness.

## Backend and performance-sensitive certification

Evidence is backend-specific when the contract is backend-specific.

- CPU correctness does not certify CUDA device behavior.
- Eager execution does not by itself certify a compiled execution contract.
- A skipped backend test is unexecuted evidence, not a pass.
- When a change affects compilation, device placement, fixed-shape execution, or another backend-sensitive property, run the matching certification before claiming that property.
- Performance-sensitive designs should be tested/benchmarked at the level needed to catch an algorithmic regression; do not turn fragile timing thresholds into ordinary unit tests without a clear reason.

## Packaging and environment evidence

Editable source-tree tests cannot prove that the distribution is packaged correctly. Repository verification should separately establish that:

- source compiles/imports in the supported environment;
- the distribution can be built;
- the built wheel can be installed;
- the installed package exposes the expected public surface outside the source checkout.

## Failure quality

A failing test should make the violated contract discoverable. Avoid suites where many failures are merely downstream consequences of one opaque fixture or overcentralized harness.

When repeated tests require extensive implementation-specific setup, reconsider the production boundary or test abstraction rather than normalizing the complexity.
