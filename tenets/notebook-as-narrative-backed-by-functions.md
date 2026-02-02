---
id: "notebook-as-narrative-backed-by-functions"
title: "Notebook as narrative, backed by reusable functions"
status: "Active"
created: "2026-02-02"
last_reviewed: "2026-02-02"
review_frequency: "Quarterly"
tags:
  - "ux"
  - "maintainability"
---

# Tenet: notebook-as-narrative-backed-by-functions

**Title**: Notebook as narrative, backed by reusable functions

**Description**: Notebooks should communicate the scientific narrative (motivation, choices, results), while the computational core lives in reusable functions that can be tested and reused across notebooks.

**Rationale**: The repo’s value is partly explanatory. Keeping “narrative” separate from “machinery” makes the work easier to extend (new datasets, new diagnostics, new comparisons) without duplicating logic.

**Examples**:

- When a code cell becomes “library-like”, promote it into a function/module and call it from the notebook.
- Keep notebook outputs readable: clear tables, named plots, and minimal hidden state.

