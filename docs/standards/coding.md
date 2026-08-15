# Coding standards

These standards constrain LatentSeq development independently of the current implementation. They are derived from the project engineering requirements in the Paper 2 specification. A current module, test, or API is not evidence that these standards should be weakened to fit it.

## Quality target

Code, tests, and documentation must seek a near-optimal balance of five properties:

- **Correct** — solves the intended problem and respects the product contract.
- **Maintainable** — responsibilities and reasoning can be understood and changed without reconstructing the whole system.
- **Concise** — no unnecessary abstraction, ceremony, defensive code, or duplication; additional structure is justified when it reduces total comprehension cost.
- **Fast** — uses algorithms and execution patterns appropriate to the workload rather than accepting avoidable bottlenecks.
- **Effective** — uses the right representation and algorithm for the problem rather than merely producing an output that passes tests.

These properties are in tension. They are evaluated together, not as sequential boxes to check. Passing tests is evidence of correctness, not evidence that the implementation is finished.

## Design and responsibility boundaries

- Components should have one coherent responsibility and a boundary that can be explained without reference to incidental implementation details.
- Major refactors must remain cheap. If a likely design correction would require widespread mechanical surgery, the current boundaries should be reconsidered.
- Shared behavior belongs behind an explicit abstraction rather than being duplicated across consumers.
- Orchestration must not absorb responsibilities owned by the components it coordinates.
- Dependency injection is preferred for replaceable collaborators and behavior that must be isolated in tests.
- Construction logic should remain separable from runtime behavior. Public class factories may provide the user-facing entry point while delegating substantial construction to focused helpers/factories.
- Do not add abstraction merely because a concept can be named. Abstractions must reduce uncertainty, duplication, coupling, or cost of change.

## Configuration and failure behavior

- Required configuration is required. Do not use hidden `.get(...)` fallbacks or undocumented defaults for specified fields.
- Validate semantic constraints at the boundary that owns them. Do not scatter redundant guards throughout downstream code.
- Do not silently repair invalid configuration or substitute fallback behavior for a violated invariant unless the product specification explicitly requires it.
- Prefer failures that identify the violated contract. Ordinary tensor/type/shape failures may propagate naturally when additional validation would only duplicate the framework contract.
- Runtime recovery behavior must be explicit. A retry/rollback path must not silently change unrelated state or semantics.

## Documentation in code

### Module docstrings

Every module must begin with useful system context:

1. where the module participates in the larger system;
2. what classes/functions it contributes;
3. the assumptions on important data structures crossing its boundary.

A module docstring is a context and feature index, not a duplicate of every class/function docstring. If the module's contribution cannot be stated clearly, reconsider the module boundary.

### Classes and public functions

Class and public-function docstrings must make it possible to understand the public contract without reading the implementation. Class documentation should establish, in order:

1. why the class exists;
2. what it owns;
3. how it fulfills that role at a high level.

Method-level details remain with the methods rather than being redundantly imported into the class overview.

### Comments

Comments should preserve reasoning that is not obvious from the code: invariants, representation choices, algorithmic constraints, and why a non-obvious implementation exists. Comments that merely narrate syntax add noise.

It must not become impossible to infer why code is structured a certain way.

## Implementation discipline

- Type hints should make contracts easier to understand rather than being decorative.
- Keep tensor/device behavior explicit at boundaries where it matters.
- Avoid tensor-dependent Python control flow in code intended for compiled/batched execution unless the design explicitly places that control on the host.
- Prefer framework capabilities over local reimplementations when the framework already owns the behavior correctly.
- Preserve the distinction between semantic/static construction and mutable/runtime execution when the architecture relies on it.
- Do not let convenience APIs hide important user-owned behavior or make independent components impossible to use independently.

## Refactor checkpoint

After a behavior is implemented and tests pass, reevaluate the design before considering the work complete:

- Is the responsibility in the right place?
- Did the implementation become more complex than the contract requires?
- Is there duplicated knowledge?
- Would the most likely major correction be cheap?
- Are comments/docstrings explaining the right things rather than compensating for poor structure?
- Is performance appropriate to the intended backend?

If the answer exposes a structural problem, refactor from the design boundary rather than accumulating patches around the first implementation.
