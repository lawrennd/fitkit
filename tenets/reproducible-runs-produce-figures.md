---
id: "reproducible-runs-produce-figures"
title: "Reproducible runs that regenerate key outputs"
status: "Active"
created: "2026-02-02"
last_reviewed: "2026-02-02"
review_frequency: "Quarterly"
tags:
  - "reproducibility"
  - "workflow"
---

# Tenet: reproducible-runs-produce-figures

**Title**: Reproducible runs that regenerate key outputs

**Description**: The scientific software should make it easy to re-run analyses and regenerate the key outputs (tables/plots/figures) from the same inputs.

**Rationale**: This repo supports a research narrative. Readers (and future us) should be able to reproduce results without reverse-engineering ad-hoc state from a notebook session.

**Examples**:

- Prefer deterministic defaults (fixed seeds where randomness exists).
- Record key run settings (data source, thresholds, normalisation choices) alongside outputs.

