---
id: "0005"
title: "Continuous integration verifies code quality and scientific-software correctness"
status: "Proposed"
priority: "High"
created: "2026-02-02"
last_updated: "2026-02-02"
related_tenets:
  - "reproducible-runs-produce-figures"
  - "explicit-boundaries-for-scientific-software"
stakeholders:
  - "Neil Lawrennd"
tags:
  - "ci-cd"
  - "testing"
  - "automation"
---

# REQ-0005: Continuous integration verifies code quality and scientific-software correctness

## Description

The project should use continuous integration (CI) to automatically verify that changes maintain code quality, pass tests, and conform to governance standards. CI runs should be fast, deterministic, and require no cloud credentials or network access.

## Acceptance Criteria

- [ ] There is a CI configuration file (e.g. `.github/workflows/ci.yml`) that runs on every push and pull request.
- [ ] CI runs the test suite (via `pytest`) using only synthetic fixtures (no BigQuery, no cloud credentials).
- [ ] CI runs the VibeSafe structure validator to detect governance drift.
- [ ] CI runs code quality checks (linting, type checking) to maintain consistency.
- [ ] CI verifies that notebooks can be parsed (but does not require full execution with cloud resources).
- [ ] CI completes in a reasonable time (target: under 5 minutes for typical runs).
- [ ] CI status is visible on pull requests and blocks merging if checks fail (or provides clear warnings).

## Notes (Optional)

CI is a forcing function for REQ-0002 (offline testing) and REQ-0003 (algorithm/data separation). If algorithms are properly decoupled from data loading, CI can verify correctness without cloud access.

The scope here is **verification**, not deployment. We are not deploying artifacts or running production pipelines; we are checking that scientific code remains correct and maintainable.

## References

- **Related Tenets**: reproducible-runs-produce-figures; explicit-boundaries-for-scientific-software
- **Related Requirements**: REQ-0002 (testing), REQ-0003 (separation)
