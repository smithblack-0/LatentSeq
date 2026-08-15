# Contributing

LatentSeq separates engineering rules from repository-specific navigation. Read the standards before using the current code/tests as examples; the implementation is expected to change when it violates those standards.

Start with `docs/README.md` for the documentation hierarchy.

## Standards

- `docs/standards/coding.md` — what acceptable code/architecture/documentation looks like.
- `docs/standards/testing.md` — TDD and evidence requirements.
- `docs/standards/review.md` — what must be present before a change is reasonably approvable.

## Current repository guides

- `docs/guides/system.md` — architecture and source responsibility map.
- `docs/guides/testing.md` — current tests, commands, markers, and CI layout.
- `docs/guides/usage.md` — user-facing API examples.
- `CHANGELOG.md` — durable change history.

Install development dependencies with:

```bash
python -m pip install -e '.[dev]'
```

Run the CPU-certifiable suite with:

```bash
python -m pytest -m 'not cuda'
```

A change should update guides when the repository layout/usage changes and update `CHANGELOG.md` when behavior or maintainer-relevant repository capability changes. Standards should change only when the engineering policy itself is deliberately changed, never merely to fit a finished implementation.
