---
id: "0003"
title: "Set up continuous integration with GitHub Actions"
status: "Accepted"
created: "2026-02-02"
last_updated: "2026-02-02"
author: "Neil Lawrennd"
compressed: false
related_requirements:
  - "0005"
  - "0002"
tags:
  - "ci-cd"
  - "github-actions"
  - "automation"
---

# CIP-0003: Set up continuous integration with GitHub Actions

> **Note**: This CIP describes HOW to achieve REQ-0005 (continuous integration).

## Status

- [x] Proposed
- [x] Accepted
- [x] In Progress
- [x] Implemented
- [ ] Closed

## Summary

Set up GitHub Actions to automatically run tests, VibeSafe validation, and code quality checks on every push and pull request. CI will verify that the scientific software remains correct and maintainable without requiring cloud credentials.

## Motivation

Manual verification is error-prone and slows down iteration. Automated CI provides:
- Early detection of regressions (failing tests, broken imports).
- Enforcement of governance standards (VibeSafe validation).
- Consistent code quality (linting, type checking).
- Confidence that changes are safe to merge.

CI is especially important for scientific software, where subtle numerical errors can go unnoticed without systematic testing.

## Detailed Description

### CI Workflow Structure

Create `.github/workflows/ci.yml` with a single workflow that runs on:
- Every push to any branch
- Every pull request targeting `main`

The workflow will have the following jobs:

**Job 1: Tests**
- Set up Python 3.11+ (or latest stable version)
- Install fitkit package with test dependencies: `pip install -e ".[test]"`
- Run pytest: `pytest tests/ -v --tb=short`
- Use synthetic fixtures only (no BigQuery, no cloud credentials)
- Fail if any test fails

**Job 2: VibeSafe Validation**
- Set up Python 3.11+
- Install VibeSafe validator dependencies: `pip install python-frontmatter pyyaml`
- Run validator: `python scripts/validate_vibesafe_structure.py --strict`
- Fail if validation errors or warnings are detected

**Job 3: Code Quality**
- Set up Python 3.11+
- Install code quality tools: `pip install ruff mypy`
- Run linter: `ruff check fitkit/ tests/`
- Run type checker: `mypy fitkit/ --strict` (or `--check-untyped-defs` for gradual typing)
- Fail if linting errors or type errors are detected

**Job 4: Notebook Check** (optional, can defer to later)
- Install nbconvert or nbformat: `pip install nbformat`
- Verify notebooks can be parsed: `python -c "import nbformat; nbformat.read('wikipedia_editing_fitness_complexity.ipynb', as_version=4)"`
- Does NOT execute notebooks (execution requires BigQuery auth)
- Fail if notebooks are malformed

### Configuration Details

**pyproject.toml additions**:

```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pytest-cov",
]
dev = [
    "ruff",
    "mypy",
    "nbformat",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "PT"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Start permissive, tighten over time
```

**GitHub Actions workflow** (`.github/workflows/ci.yml`):

```yaml
name: CI

on:
  push:
    branches: ["*"]
  pull_request:
    branches: ["main"]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[test]"
      - name: Run tests
        run: pytest tests/ -v --tb=short

  vibesafe:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install VibeSafe dependencies
        run: pip install python-frontmatter pyyaml
      - name: Run VibeSafe validator
        run: python scripts/validate_vibesafe_structure.py --strict

  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install quality tools
        run: pip install ruff mypy
      - name: Run ruff
        run: ruff check fitkit/ tests/
      - name: Run mypy
        run: mypy fitkit/ --check-untyped-defs

  notebook:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install nbformat
        run: pip install nbformat
      - name: Check notebook format
        run: |
          python -c "import nbformat; nbformat.read('wikipedia_editing_fitness_complexity.ipynb', as_version=4)"
```

### Non-Goals

- **No deployment**: CI verifies correctness but does not deploy artifacts (no PyPI publishing, no Docker images, no cloud deployments).
- **No expensive checks**: CI does not run full notebook execution (requires BigQuery) or large-scale benchmarks.
- **No cloud credentials**: CI must work entirely offline using synthetic fixtures.

## Implementation Plan

1. **Add CI workflow file**:
   - Create `.github/workflows/ci.yml` with the structure above.
   - Start with test and VibeSafe jobs; add quality/notebook jobs incrementally.

2. **Update pyproject.toml**:
   - Add `[project.optional-dependencies]` for `test` and `dev` extras.
   - Add `[tool.pytest.ini_options]`, `[tool.ruff]`, and `[tool.mypy]` configs.

3. **Ensure tests pass offline**:
   - Verify that `pytest tests/` runs successfully without network or cloud credentials.
   - Fix any tests that inadvertently depend on external resources.

4. **Commit and push**:
   - Commit CI workflow and config changes.
   - Push to a branch and verify that GitHub Actions runs successfully.
   - Iterate on any failures (install issues, path problems, etc.).

5. **Enable branch protection** (optional):
   - In GitHub repo settings, require CI checks to pass before merging to `main`.

## Backward Compatibility

No backward compatibility concerns. This is a new infrastructure addition that does not change any code APIs or data formats.

## Testing Strategy

The CI system itself is tested by:
- Pushing a branch and verifying that all jobs run and pass.
- Intentionally introducing a failing test, linting error, or governance issue to verify that CI catches it.

## Related Requirements

This CIP implements:
- **REQ-0005**: Continuous integration verifies code quality and correctness.
- **REQ-0002**: Testing (CI enforces that tests exist and pass).

## Implementation Status

- [ ] Create `.github/workflows/ci.yml`
- [ ] Update `pyproject.toml` with test/dev dependencies and tool configs
- [ ] Verify CI runs successfully on a test branch
- [ ] Enable branch protection (optional)

## References

- GitHub Actions docs: https://docs.github.com/en/actions
- pytest docs: https://docs.pytest.org/
- ruff docs: https://docs.astral.sh/ruff/
