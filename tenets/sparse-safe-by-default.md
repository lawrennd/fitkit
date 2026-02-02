---
id: "sparse-safe-by-default"
title: "Sparse-safe by default"
status: "Active"
created: "2026-02-02"
last_reviewed: "2026-02-02"
review_frequency: "Quarterly"
tags:
  - "performance"
  - "scalability"
---

# Tenet: sparse-safe-by-default

**Title**: Sparse-safe by default

**Description**: Core computations should operate efficiently on sparse matrices and avoid accidental densification.

**Rationale**: The kinds of bipartite incidence matrices we work with can be large and sparse. If the “happy path” densifies, the software won’t scale beyond toy examples.

**Examples**:

- Keep data in sparse formats end-to-end where possible.
- Prefer algorithms/implementations that can compute diagnostics without materializing full dense matrices.

