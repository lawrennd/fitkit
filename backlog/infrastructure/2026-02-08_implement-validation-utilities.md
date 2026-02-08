---
id: "2026-02-08_implement-validation-utilities"
title: "Implement statistical validation utilities"
status: "Proposed"
priority: "High"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "infrastructure"
related_cips: ["0006"]
owner: "Neil Lawrence"
dependencies: ["2026-02-08_create-community-module-structure"]
tags:
- backlog
- community-detection
- validation
- statistical-testing
---

# Task: Implement Statistical Validation Utilities

> **Note**: Backlog tasks are DOING the work defined in CIPs (HOW).  
> Use `related_cips` to link to CIPs. Don't link directly to requirements (bottom-up pattern).

## Description

Implement statistical validation methods for community detection in `fitkit/community/validation.py`. These methods provide rigorous hypothesis tests and bounds for community structure quality.

## Acceptance Criteria

- [ ] **Permutation test for eigengaps**:
  - `validate_eigengap(adjacency_matrix, n_shuffles=100) -> (gap, p_value)`
  - Shuffle adjacency edges, recompute Laplacian eigenvalues
  - Null distribution: λ₂/λ₃ ratios from random networks
  - Return observed gap and p-value (fraction of shuffles with larger gap)

- [ ] **Cheeger conductance bounds**:
  - `compute_cheeger_bound(adjacency_matrix, community_labels) -> dict`
  - For each community k: φₖ = cut(Sₖ, Sₖᶜ) / min(vol(Sₖ), vol(Sₖᶜ))
  - Link to spectral gap via Cheeger inequality: λ₂ ≥ φ²/2
  - Return {community_k: {'conductance': φₖ, 'lower_bound': φₖ²/2}}

- [ ] **Configuration model null for bipartite**:
  - `validate_bipartite_structure(M, n_shuffles=100) -> (statistic, p_value)`
  - Preserve row/column sums, shuffle connections
  - Null distribution: spectral gaps from degree-preserving random graphs
  - Return observed statistic and p-value

- [ ] **Effective rank diagnostic**:
  - `compute_effective_rank(transition_matrix, t_max=10) -> R_values`
  - R(t) = exp(H(pₜ)) where H(pₜ) = -∑ pᵢ(t) log pᵢ(t)
  - Compute via matrix powers: pₜ = Pᵗp₀
  - Return time series R(t) for t=1...t_max
  - Data-driven alternative to fixed thresholds

- [ ] All functions handle both dense and sparse matrices
- [ ] All functions have comprehensive docstrings with references
- [ ] All functions expose random seed parameter for reproducibility

## Implementation Notes

**Permutation strategy**:
- For bipartite networks, only permute within row-column structure (preserve bipartiteness)
- For general graphs, use edge swapping to preserve degree sequence
- Use scipy.sparse operations for efficiency

**Cheeger inequality** (rigorous bound on community quality):
- λ₂ ≥ φ²/2 ≥ λ₂/2
- φ (conductance) measures "bottleneck" of community
- Small φ → strong community isolation

**Effective rank interpretation**:
- R(t) << n: strong community structure (few effective states)
- R(t) ≈ n: weak structure (entropy diffuses uniformly)
- Elbow in R(t) curve indicates natural timescale for communities

**References**:
- Cheeger bound: Chung (1997) Spectral Graph Theory
- Effective rank: Wang et al. (2017) ICML
- Configuration model: Newman (2018) Networks

## Related

- CIP: 0006
- Links to CIP-0006 validation methods section

## Progress Updates

### 2026-02-08

Task created from CIP-0006 acceptance.
