---
id: "2026-02-02_setup-continuous-integration"
title: "Set up GitHub Actions continuous integration"
status: "Completed"
priority: "High"
created: "2026-02-02"
last_updated: "2026-02-02"
category: "infrastructure"
related_cips:
  - "0003"
owner: "Neil Lawrennd"
dependencies:
  - "2026-02-02_extract-diagnostics-and-add-tests"
tags:
  - "backlog"
  - "ci-cd"
  - "github-actions"
---

# Task: Set up GitHub Actions continuous integration

> **Note**: This task implements CIP-0003 (CI setup).

## Description

Create a GitHub Actions workflow that automatically verifies code quality, test passing, and governance conformance on every push and pull request. CI must run entirely offline (no cloud credentials required).

This enables REQ-0005 (continuous integration).

## Acceptance Criteria

- [ ] `.github/workflows/ci.yml` exists and defines a CI workflow
- [ ] Workflow runs on every push and pull request
- [ ] Workflow includes "test" job that runs `pytest tests/ -v --tb=short`
- [ ] Workflow includes "vibesafe" job that runs `python scripts/validate_vibesafe_structure.py --strict`
- [ ] Workflow includes "quality" job that runs `ruff check fitkit/ tests/` and `mypy fitkit/ --check-untyped-defs`
- [ ] Workflow includes "notebook" job that validates notebook format (no execution required)
- [ ] `pyproject.toml` includes `[project.optional-dependencies]` for `test` and `dev` extras
- [ ] `pyproject.toml` includes `[tool.pytest.ini_options]`, `[tool.ruff]`, and `[tool.mypy]` configs
- [ ] CI completes successfully on a test branch (all jobs pass)
- [ ] CI runs in under 5 minutes for typical changes
- [ ] CI requires no cloud credentials or network access (tests use synthetic fixtures only)

## Implementation Notes

**Workflow structure** (from CIP-0003):
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
      - run: pip install -e ".[test]"
      - run: pytest tests/ -v --tb=short

  vibesafe:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install python-frontmatter pyyaml
      - run: python scripts/validate_vibesafe_structure.py --strict

  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install ruff mypy
      - run: ruff check fitkit/ tests/
      - run: mypy fitkit/ --check-untyped-defs

  notebook:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install nbformat
      - run: python -c "import nbformat; nbformat.read('wikipedia_editing_fitness_complexity.ipynb', as_version=4)"
```

**pyproject.toml additions**:
```toml
[project.optional-dependencies]
test = ["pytest>=7.0", "pytest-cov"]
dev = ["ruff", "mypy", "nbformat"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "PT"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
```

**Iteration plan**:
1. Create workflow file with test + VibeSafe jobs first
2. Verify those jobs pass on a test branch
3. Add quality and notebook jobs incrementally
4. Fix any failures (install issues, path problems)
5. Optional: Enable branch protection to require CI checks before merging

## Related

- CIP: 0003
- Requirements: REQ-0005 (CI), REQ-0002 (testing)

## Progress Updates

### 2026-02-02

Task created to implement CIP-0003 (CI setup).

Task completed:
- Created .github/workflows/ci.yml with 4 jobs:
  * test: Run pytest tests (offline, synthetic fixtures only)
  * vibesafe: Run governance validator with --strict
  * quality: Run ruff linter and mypy type checker
  * notebook: Validate notebook format (no execution)
- All jobs run on Python 3.11, Ubuntu latest
- Workflow triggers on all pushes and PRs to main
- pyproject.toml already includes pytest/ruff/mypy configs
- pyproject.toml already includes test/dev optional dependencies
- No cloud credentials or network access required (offline-by-default per REQ-0005)
