---
id: "0004"
title: "A stable data-loading adapter interface exists for future Lynguine integration"
status: "Proposed"
priority: "Medium"
created: "2026-02-02"
last_updated: "2026-02-02"
related_tenets:
  - "integration-friendly-data-loading"
  - "explicit-boundaries-for-scientific-software"
stakeholders:
  - "Neil Lawrennd"
tags:
  - "architecture"
  - "adapter"
  - "doa"
---

# REQ-0004: A stable data-loading adapter interface exists for future Lynguine integration

## Description

Fitkit should define and document a small, stable interface for data-loading adapters so that, over time, data acquisition can be delegated to a data-oriented access layer (e.g. `~/lawrennd/lynguine`) without changing algorithm code.

For now, the implementation may live locally in this repository. The important outcome is that the boundary is explicit and future integration is incremental.

## Acceptance Criteria

- [ ] The project defines a documented loader interface (inputs, outputs, and required metadata) used by analyses/notebooks.
- [ ] At least one local loader conforms to the interface (e.g. “Wikipedia edits via BigQuery or cache”), and at least one synthetic/offline loader exists for tests.
- [ ] The interface is backend-agnostic: it does not require BigQuery-specific types/clients in the algorithm layer.
- [ ] The interface includes provenance/caching metadata sufficient to reproduce a run (source identifier, cache key/path, query parameters, thresholds).
- [ ] There is an explicit statement of non-goals: we are not adopting the full Lynguine access/assess/address stack inside fitkit at this stage; fitkit remains a scientific codebase with a locally defined adapter boundary.

## Notes (Optional)

This requirement targets compatibility with a Lynguine-style DOA, where access (loading) is separate from assessment/compute. It does not prescribe the final integration mechanism.

## References

- **Related Tenets**: integration-friendly-data-loading; explicit-boundaries-for-scientific-software

