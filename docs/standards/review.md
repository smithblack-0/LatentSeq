# Review standards

Review is an independent engineering check, not a confirmation that the author ran tests. A change must be presented so a reviewer can understand what changed, what contracts it claims to satisfy, what evidence supports those claims, and what remains uncertain.

## Review must be possible at the scale of the change

The larger the change, the more explicitly it must be decomposed for review.

A substantial pull request must provide:

- an honest statement of scope and magnitude;
- a broken-down changelist organized by responsibility/subsystem rather than raw filenames;
- the intended contract or behavior for each major change area;
- the primary implementation location and evidence for each area when that materially reduces review cost;
- known risks, unverified backend requirements, and intentional omissions;
- validation actually performed, distinguished from validation merely planned or skipped.

A large diff summarized as one feature plus “tests pass” is not reviewable enough to approve.

## Review the design, not only the diff

For each major responsibility, review asks whether the change is:

- correct for the intended product behavior;
- placed behind the right boundary;
- maintainable under plausible future refactors;
- concise rather than ceremony-heavy or duplicated;
- fast/effective for the intended workload and backend.

Tests are evidence for these judgments, not substitutes for them.

Reviewers should explicitly challenge:

- responsibilities that moved or became hidden;
- new abstractions with unclear payoff;
- duplicated knowledge or configuration;
- silent defaults or repairs;
- implementation-derived contracts that are not grounded in the specification/approved design;
- documentation that rationalizes the finished implementation instead of stating an independent requirement;
- claims of backend certification without backend-specific evidence.

## Changelists and evidence maps

A changelist answers **what changed**. An evidence map answers **where should I inspect it and what verifies it**. For small changes they may be trivial; for large changes they should be explicit.

The durable `CHANGELOG.md` records user-visible and maintainer-relevant changes over time. Pull-request metadata may link to it, but must still describe the current review scope clearly enough that the reviewer does not have to reconstruct the PR from repository history.

## Pull-request metadata

PR title and description are part of the engineering deliverable.

- The title must reflect the actual scope, not make a repository-scale build sound like a small feature.
- The description should lead with the change and its consequences, not a secondary implementation curiosity.
- Major subsystems should be listed at a granularity suitable for independent review.
- Test counts alone are insufficient; identify the kinds of evidence and any unexecuted requirements.
- If the PR contains repository/productization work as well as product code, state both.

Logical commits are encouraged when they let a reviewer inspect separable concerns independently. Commit structure must not be used to hide that the resulting system still has cross-cutting risks.

## Approval threshold

Do not approve a change merely because:

- the suite is green;
- the code looks plausible in aggregate;
- the PR description claims the specification was implemented;
- only one backend was available;
- the reviewer cannot identify a concrete bug in an otherwise opaque change.

Approval requires enough decomposition and evidence to make silent major errors reasonably discoverable.

Known gaps may be accepted when they are explicitly scoped and appropriate for the stage of development. They must not be silently converted into successful validation.
