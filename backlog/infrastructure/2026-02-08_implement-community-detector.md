---
id: "2026-02-08_implement-community-detector"
title: "Implement CommunityDetector class with iterative algorithm"
status: "Completed"
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
- sklearn-interface
---

# Task: Implement CommunityDetector Class with Iterative Algorithm

> **Note**: Backlog tasks are DOING the work defined in CIPs (HOW).  
> Use `related_cips` to link to CIPs. Don't link directly to requirements (bottom-up pattern).

## Description

Implement the core `CommunityDetector` class following the iterative algorithm from Sanguinetti, Lawrence & Laidler (2005) with sklearn-style interface.

**Core algorithm**: Iteratively add eigenvectors (q = 2, 3, ...) until origin-detector cluster remains empty, using elongated k-means that exploits radial structure of insufficient-dimensional projections.

## Acceptance Criteria

- [x] `CommunityDetector` class in `fitkit/community/detection.py`
- [x] Sklearn-style methods: `__init__()`, `fit()`, `fit_predict()`
- [x] Parameters: `method='iterative'`, `n_communities='auto'`, `max_communities=8`, `lambda_elongation=0.2`
- [x] Implements iterative eigenvector algorithm:
  - Start with q=2 eigenvectors
  - Initialize q centers + 1 at origin
  - Run elongated k-means
  - If origin captures points, increment q and repeat
  - Terminate when origin cluster is empty or reaches max
- [x] Implements elongated k-means with Mahalanobis distance:
  - For centers not at origin: d²(x, c) = (x-c)ᵀM(x-c) where M adapts along radial direction
  - For center at origin: use Euclidean distance
- [x] Exposes attributes after fitting:
  - `labels_` - community assignments
  - `n_communities_` - number detected
  - `eigenvalues_` - for diagnostics
  - `n_iterations_` - iteration history
- [x] Handles edge cases: small networks, no structure, convergence failures
- [x] Works on both dense and sparse matrices

## Implementation Notes

**Elongated distance metric** (from 2005 paper):

M = (1/λ)(I - ccᵀ/||c||²) + λ(ccᵀ/||c||²)

- λ = lambda_elongation parameter (default 0.2)
- Small λ → strong radial elongation
- Downweights distances along radial direction c
- Penalizes distances perpendicular to c

**Initialization strategy** (critical for convergence):
- First q centers: select from detected clusters at previous iteration
- (q+1)-th center: always at origin
- Use farthest point from origin for first center
- Use point maximizing ||x|| while minimizing dot product with existing centers

**Reference implementation**: https://github.com/lawrennd/spectral/blob/master/matlab/SpectralCluster.m

## Related

- CIP: 0006
- Algorithm: Sanguinetti, Lawrence & Laidler (2005)
- Reference code: https://github.com/lawrennd/spectral

## Progress Updates

### 2026-02-08

Task created from CIP-0006 acceptance.

Implementation completed. Full iterative algorithm implemented with:
- Eigenvector computation from bipartite transition matrix
- Iterative algorithm with origin-detector validation
- Elongated k-means with Mahalanobis-like distance metric
- Minimum cluster size check (5% of samples) for robustness
- Support for both sparse and dense matrices
- Edge case handling for small networks
- Comprehensive docstrings

**Current Status (2026-02-08):**

All major fixes completed based on MATLAB reference (~/lawrennd/spectral/matlab/SpectralCluster.m):
- ✓ Transition matrix: T = D_c^{-1} M D_p^{-1} M^T  
- ✓ Elongated distance: d² = λ*||radial||² + (1/λ)*||tangential||²
- ✓ Projection calculation: fixed broadcasting (was using np.outer incorrectly)
- ✓ Warm-start: extend all previous centers correctly
- ✓ Origin center: allowed to move during k-means convergence
- ✓ **CRITICAL**: Include trivial eigenvector (eigenvectors[:, 0:q] not [:, 1:q+1])

**Root Cause Identified:**
MATLAB SpectralCluster.m uses `PcEig = Y(:,(1:Dim))` which includes the trivial eigenvector (eigenvalue ≈1) as the first column. Python implementation was incorrectly skipping it with `eigenvectors[:, 1:q+1]`. The trivial eigenvector provides essential scale/mass distribution information that the origin detector needs.

**Validation Results:**
Comprehensive test suite confirms perfect detection:
- Perfect 2-block bipartite: ✓ 2 communities, 100% purity
- Modular with cross-connections: ✓ 2 communities, 100% purity  
- Perfect 3-block bipartite: ✓ 3 communities, 100% purity

**Status: Implementation COMPLETED and validated against test networks.**
