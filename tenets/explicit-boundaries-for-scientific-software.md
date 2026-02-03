---
id: "explicit-boundaries-for-scientific-software"
title: "Explicit boundaries between data access and algorithms"
status: "Active"
created: "2026-02-02"
last_reviewed: "2026-02-02"
review_frequency: "Quarterly"
tags:
  - "architecture"
  - "scientific-software"
  - "separation-of-concerns"
---

# Tenet: explicit-boundaries-for-scientific-software

**Title**: Explicit boundaries between data access and algorithms

**Description**: Keep a clear boundary between *how data is acquired/loaded* and *how indices/diagnostics are computed*, so scientific iteration remains reproducible, testable, and comparable across datasets.

**Rationale**: Scientific software evolves quickly (new datasets, new preprocessing, alternative baselines). If data loading and algorithms are coupled, it becomes hard to test changes, hard to compare runs, and easy to accidentally change “the experiment” while changing code.

**Examples**:

- Algorithm code should accept in-memory inputs (matrices/labels/metadata) and never reach into BigQuery/filesystems directly.
- Data loading code may change frequently, but should not force changes in algorithm APIs.

