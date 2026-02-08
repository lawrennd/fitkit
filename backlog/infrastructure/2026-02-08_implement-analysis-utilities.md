---
id: "2026-02-08_implement-analysis-utilities"
title: "Implement within-community analysis utilities"
status: "Proposed"
priority: "Medium"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "infrastructure"
related_cips: ["0006"]
owner: "Neil Lawrence"
dependencies: ["2026-02-08_implement-community-detector"]
tags:
- backlog
- community-detection
- analysis
---

# Task: Implement Within-Community Analysis Utilities

> **Note**: Backlog tasks are DOING the work defined in CIPs (HOW).  
> Use `related_cips` to link to CIPs. Don't link directly to requirements (bottom-up pattern).

## Description

Implement utilities for analyzing economic fitness and complexity measures within detected communities in `fitkit/community/analysis.py`.

## Acceptance Criteria

- [ ] `within_community_analysis(M, community_labels, eci_estimator, fitness_estimator) -> dict`
  - Extracts subnetworks for each community
  - Runs ECI and Fitness-Complexity on each subnetwork
  - Computes correlations (Pearson, Spearman) between measures
  - Returns structured results for each community

- [ ] `compare_global_vs_local(M, community_labels, eci_estimator, fitness_estimator) -> dict`
  - Computes global ECI/Fitness on full network
  - Computes within-community measures for each subnetwork
  - Compares global vs. within-community correlations
  - Returns comparison metrics

- [ ] Both functions:
  - Handle sparse and dense matrices
  - Work with sklearn-style estimators (ECI, FitnessComplexity)
  - Return structured dictionaries with clear keys
  - Include metadata (network sizes, community sizes)

- [ ] Output format example:
  ```python
  {
    'global': {
      'eci_fitness_corr': 0.85,
      'eci_complexity_corr': -0.45,
      ...
    },
    'communities': {
      0: {
        'n_countries': 25,
        'n_products': 120,
        'eci_fitness_corr': 0.92,
        'eci_complexity_corr': -0.35,
        ...
      },
      1: {...},
      ...
    },
    'improvement': {
      'eci_fitness': 0.07,  # local - global
      ...
    }
  }
  ```

## Implementation Notes

**Subnetwork extraction**:
- For bipartite networks, preserve bipartite structure
- Filter M to rows/columns in community
- Handle edge case: communities too small for meaningful correlation

**Correlation computation**:
- Pearson for linear relationships
- Spearman for rank-order agreement
- Report both to capture different aspects

**Design rationale** (from CIP-0006):
These utilities test the "Morphology B complementarity" hypothesis from economic-fitness.tex: modular networks should show higher within-community correlations than global correlations, because communities are more structurally homogeneous.

## Related

- CIP: 0006
- Depends on: `CommunityDetector` for community labels
- Uses: `fitkit.algorithms.ECI`, `fitkit.algorithms.FitnessComplexity`

## Progress Updates

### 2026-02-08

Task created from CIP-0006 acceptance.
