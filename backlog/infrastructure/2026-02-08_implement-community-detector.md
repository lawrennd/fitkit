---
id: "2026-02-08_implement-community-detector"
title: "Implement CommunityDetector class with iterative algorithm"
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
- sklearn-interface
---

# Task: Implement CommunityDetector Class with Iterative Algorithm

> **Note**: Backlog tasks are DOING the work defined in CIPs (HOW).  
> Use `related_cips` to link to CIPs. Don't link directly to requirements (bottom-up pattern).

## Description

Implement the core `CommunityDetector` class following the iterative algorithm from Sanguinetti, Lawrence & Laidler (2005) with sklearn-style interface.

**Core algorithm**: Iteratively add eigenvectors (q = 2, 3, ...) until origin-detector cluster remains empty, using elongated k-means that exploits radial structure of insufficient-dimensional projections.

## Acceptance Criteria

- [ ] `CommunityDetector` class in `fitkit/community/detection.py`
- [ ] Sklearn-style methods: `__init__()`, `fit()`, `fit_predict()`
- [ ] Parameters: `method='iterative'`, `n_communities='auto'`, `max_communities=8`, `lambda_elongation=0.2`
- [ ] Implements iterative eigenvector algorithm:
  - Start with q=2 eigenvectors
  - Initialize q centers + 1 at origin
  - Run elongated k-means
  - If origin captures points, increment q and repeat
  - Terminate when origin cluster is empty
- [ ] Implements elongated k-means with Mahalanobis distance:
  - For centers not at origin: d²(x, c) = (x-c)ᵀM(x-c) where M adapts along radial direction
  - For center at origin: use Euclidean distance
- [ ] Exposes attributes after fitting:
  - `labels_` - community assignments
  - `n_communities_` - number detected
  - `eigenvalues_` - for diagnostics
  - `n_iterations_` - iteration history
- [ ] Handles edge cases: small networks, no structure, convergence failures
- [ ] Works on both dense and sparse matrices

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
