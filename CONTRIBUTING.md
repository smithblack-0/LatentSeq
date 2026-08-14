# Contributing

LatentSeq is currently in its initial implementation/review phase. Changes should remain easy to audit against the generator specification and the subsystem boundaries documented in the repository.

Start with:

- `docs/development.md` for the source/responsibility map and development commands.
- `docs/testing.md` for test ownership and backend certification.
- `docs/review-guide.md` for the subsystem-by-subsystem change map.
- `CHANGELOG.md` for the durable change inventory.

Install development dependencies with:

```bash
python -m pip install -e '.[dev]'
```

Run the CPU-certifiable suite with:

```bash
python -m pytest -m 'not cuda'
```

Behavioral changes should update the test boundary that owns the behavior and the changelog entry when the change is user-visible or materially affects maintainers. Compile/CUDA-sensitive changes require the corresponding backend checks; passing unrelated CPU tests is not sufficient evidence.
