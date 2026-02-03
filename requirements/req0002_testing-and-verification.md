---
id: "0002"
title: "Testing and verification are routine and offline-by-default"
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
  - "testing"
  - "verification"
  - "scientific-software"
---

# REQ-0002: Testing and verification are routine and offline-by-default

## Description

The project should support routine verification of core computations and invariants so changes can be made confidently. Tests must be runnable in typical development environments without requiring network access, cloud credentials, or large datasets.

This includes both numerical correctness on small fixtures and structural expectations (shapes, types, determinism, and stable handling of edge cases like empty rows/columns).

## Acceptance Criteria

- [ ] There is a single, documented command to run the tests locally.
- [ ] The default test run does not require network access or cloud credentials.
- [ ] Core algorithmic routines are covered by unit tests on small synthetic fixtures:
  - Fitness–Complexity iteration
  - masked Sinkhorn/IPF scaling
  - ECI/PCI computation
- [ ] Tests include at least one fixture designed to exercise sparsity/edge cases (e.g. isolated nodes, near-empty support).
- [ ] Test outcomes are deterministic (or document and control any randomness).

## Notes (Optional)

The scope here is verification of scientific-software behavior, not statistical validation of the underlying model assumptions.

## References

- **Related Tenets**: reproducible-runs-produce-figures; explicit-boundaries-for-scientific-software

