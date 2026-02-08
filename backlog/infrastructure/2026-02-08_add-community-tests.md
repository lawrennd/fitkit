---
id: "2026-02-08_add-community-tests"
title: "Add comprehensive tests for community detection"
status: "Proposed"
priority: "High"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "infrastructure"
related_cips: ["0006"]
owner: "Neil Lawrence"
dependencies: 
- "2026-02-08_implement-community-detector"
- "2026-02-08_implement-validation-utilities"
- "2026-02-08_implement-analysis-utilities"
tags:
- backlog
- community-detection
- testing
---

# Task: Add Comprehensive Tests for Community Detection

> **Note**: Backlog tasks are DOING the work defined in CIPs (HOW).  
> Use `related_cips` to link to CIPs. Don't link directly to requirements (bottom-up pattern).

## Description

Create comprehensive test suite for community detection functionality in `fitkit/tests/test_community.py` covering the iterative algorithm, validation methods, and analysis utilities.

## Acceptance Criteria

### CommunityDetector Tests

- [ ] **Synthetic networks with known structure**:
  - Two-block modular network → detects 2 communities
  - Three-block modular network → detects 3 communities
  - Single homogeneous block → detects 1 community (origin-detector never triggered)

- [ ] **Iterative algorithm mechanics**:
  - Test q increments correctly (q=2, 3, 4, ...)
  - Test origin-detector cluster behavior
  - Test termination when origin empty
  - Test max_communities limit

- [ ] **Elongated k-means**:
  - Verify distance metric computation
  - Test lambda_elongation parameter effect
  - Test that radial elongation helps separation

- [ ] **Sklearn interface compliance**:
  - Test `fit()`, `fit_predict()` methods
  - Test attribute exposure (`labels_`, `n_communities_`, etc.)
  - Test parameter validation

- [ ] **Edge cases**:
  - Very small networks (n < 5)
  - Networks with no clear structure
  - Convergence failures
  - Sparse vs. dense matrices

### Validation Tests

- [ ] **Permutation tests**:
  - Test p-value computation
  - Test null distribution generation
  - Test random seed reproducibility
  - Test on network with/without structure

- [ ] **Cheeger bounds**:
  - Test conductance computation
  - Verify bound inequality: λ₂ ≥ φ²/2
  - Test on communities with known quality

- [ ] **Effective rank**:
  - Test R(t) computation
  - Test entropy calculation
  - Test time evolution
  - Verify R(t) << n for strong structure

### Analysis Tests

- [ ] **Within-community analysis**:
  - Test subnetwork extraction
  - Test correlation computation
  - Test output format
  - Test edge cases (small communities)

- [ ] **Global vs. local comparison**:
  - Test on modular synthetic network
  - Verify local > global correlations for modular networks
  - Test output structure

### Integration Tests

- [ ] **Full pipeline on synthetic data**:
  - Generate modular bipartite network
  - Detect communities
  - Validate structure
  - Analyze within communities
  - Verify all components work together

- [ ] **Comparison with 2005 MATLAB reference**:
  - Test on canonical datasets from original paper
  - Verify clustering results match (allow minor differences due to initialization)

## Implementation Notes

**Test data generation**:
```python
def make_modular_bipartite(n_communities, block_size, noise_level):
    """Generate bipartite network with known community structure"""
    # Perfect block structure + noise edges
    pass
```

**Testing philosophy**:
- Unit tests for individual components
- Integration tests for full pipeline
- Property-based tests for invariants (e.g., labels should be [0, n-1])
- Regression tests against reference implementation

**Coverage goal**: >90% for all new code

## Related

- CIP: 0006
- Testing Strategy section in CIP-0006
- Reference: https://github.com/lawrennd/spectral

## Progress Updates

### 2026-02-08

Task created from CIP-0006 acceptance.
