---
id: "integration-friendly-data-loading"
title: "Integration-friendly data loading"
status: "Active"
created: "2026-02-02"
last_reviewed: "2026-02-02"
review_frequency: "Quarterly"
tags:
  - "architecture"
  - "data-loading"
  - "doa"
---

# Tenet: integration-friendly-data-loading

**Title**: Integration-friendly data loading

**Description**: Data loading should be structured so it can later integrate with a data-oriented access layer (e.g. `~/lawrennd/lynguine`) without rewriting the algorithms.

**Rationale**: Fitkit’s analysis pipelines should remain portable across environments and data backends. A future integration should be an adapter/wrapper change, not a core algorithm refactor.

**Examples**:

- Prefer a small, explicit loader interface (inputs/outputs clearly documented) over embedding backend-specific logic in algorithms.
- Keep caching and provenance explicit so external access layers can own storage policies later.

