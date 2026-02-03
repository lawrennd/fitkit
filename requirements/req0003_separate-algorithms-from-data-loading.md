---
id: "0003"
title: "Algorithms are separate from data loading and caching"
status: "Proposed"
priority: "High"
created: "2026-02-02"
last_updated: "2026-02-02"
related_tenets:
  - "explicit-boundaries-for-scientific-software"
  - "integration-friendly-data-loading"
stakeholders:
  - "Neil Lawrennd"
tags:
  - "architecture"
  - "data-loading"
  - "separation-of-concerns"
---

# REQ-0003: Algorithms are separate from data loading and caching

## Description

The codebase should enforce a clear separation between algorithmic computation (indices and diagnostics) and data acquisition/loading/caching. This enables reuse across datasets/backends and prevents accidental coupling between experimental setup and computation.

Algorithms should operate on explicit in-memory inputs (e.g. sparse incidence matrices, labels, and minimal metadata) and should not know *where those inputs came from*.

## Acceptance Criteria

- [ ] Algorithm code does not perform I/O (no direct filesystem or network access).
- [ ] Data-loading code is the only layer that performs I/O and returns an explicit in-memory representation suitable for algorithms.
- [ ] Caching behavior is explicit (paths, cache keys, invalidation/refetch criteria are recorded in configuration or run metadata).
- [ ] It is possible to swap the data source (e.g. BigQuery vs local files vs synthetic fixtures) without changing algorithm code.
- [ ] It is possible to run algorithms purely from in-memory fixtures in tests.

## References

- **Related Tenets**: explicit-boundaries-for-scientific-software; integration-friendly-data-loading

