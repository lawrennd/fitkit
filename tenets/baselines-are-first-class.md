---
id: "baselines-are-first-class"
title: "Baselines are first-class (ECI/PCI and others)"
status: "Active"
created: "2026-02-02"
last_reviewed: "2026-02-02"
review_frequency: "Quarterly"
tags:
  - "evaluation"
  - "comparisons"
---

# Tenet: baselines-are-first-class

**Title**: Baselines are first-class (ECI/PCI and others)

**Description**: The scientific software should make it natural to compare outputs against standard baselines—especially **ECI/PCI** computed on the same derived data matrix—so interpretation is anchored.

**Rationale**: The intent so far (as seen in the notebook) is not just to compute a single score, but to **situate it**: side-by-side rankings, scatterplots, and “what moved when I changed X?” checks.

**Examples**:

- Provide utilities that compute FC outputs and ECI/PCI together and return aligned tables.
- Make “Fitness vs ECI” (and the product/word analogue) a default diagnostic view.

