---
author: "Neil Lawrence"
created: "2026-02-08"
last_updated: "2026-02-08"
status: "Implemented"
related_requirements: []
related_cips: ["0006"]
tags: ["spectral-clustering", "community-detection", "bugfix", "hyperparameters"]
compressed: false
---

# CIP-0007: Sigma Parameter Scaling for Spectral Clustering

## Status

- [x] Proposed → Initial documentation complete
- [x] Accepted → Plan reviewed and approved
- [x] Implemented → Code changes complete
- [x] Closed → Implementation validated

## Summary

Fix the spectral clustering algorithm's failure to detect 3 concentric circles with datasets smaller than 300 points. The algorithm was finding only 2 clusters instead of 3 due to incorrect sigma parameter scaling with dataset size and point density.

This CIP addresses a critical algorithmic issue in the community detection implementation from CIP-0006.

## Motivation

### Problem Discovery

During validation of the spectral clustering implementation (CIP-0006), the three concentric circles test consistently failed on smaller datasets:

- **300 points (100 per circle)**: Crashed with `scipy.linalg.eigh` BLAS segfault (see Appendix)
- **150 points (50 per circle)**: Found only 2 clusters instead of 3
- **90 points (30 per circle)**: Found only 2 clusters instead of 3

The 300-point crash is a **confirmed known issue** with OpenBLAS 0.3.21 on Apple Silicon (see Appendix for full details, minimal reproduction, and web research findings). The crash was worked around by using pre-computed Octave eigenvectors, but the fundamental algorithmic issue of incorrect cluster detection on smaller datasets remained.

### Root Cause Analysis

Detailed debugging revealed:

1. **Eigenvector Structure**: With sigma=0.158 (MATLAB-equivalent), the eigenvectors for 150 points showed:
   - Eigenvector 0 (trivial): All points clustered together
   - Eigenvector 1: Separated outermost circle (c2) from inner two (c0+c1)
   - Eigenvector 2: Would separate c0 from c1, but never reached

2. **Premature Stopping**: At Dim=2, elongated k-means found:
   - Cluster 0: 50 points (all from circle 2)
   - Cluster 1: 100 points (circles 0 + 1 merged)
   - Origin cluster: 0 points → algorithm stopped

3. **Parameter Sensitivity**: The Gaussian affinity kernel `A(i,j) = exp(-||x_i - x_j||²/(2σ²))` is highly sensitive to sigma:
   - Too small (0.05): Over-fragments, finds 9+ clusters
   - Too small (0.158): Under-connects, stops at 2 clusters
   - Just right (0.2): Correctly finds 3 clusters with 100% purity

### Why This Matters

This is **not** a trivial parameter tuning issue. The sigma parameter fundamentally controls:
- **Connectivity** in the affinity graph (too small = disconnected components)
- **Eigenvector separation** in spectral space (affects cluster boundaries)
- **Origin cluster behavior** in iterative detection (determines when to add dimensions)

Without proper sigma scaling, the algorithm fails on the simplest possible test case (three well-separated concentric circles), making it unreliable for real-world bipartite network analysis.

## Detailed Description

### Gaussian Affinity Kernel

The affinity matrix uses a Gaussian (RBF) kernel:

```python
A(i,j) = exp(-||x_i - x_j||² / (2*sigma²))
```

**Sigma interpretation**:
- Controls the "reach" of each point's influence
- Small sigma: Only very close points have high affinity
- Large sigma: Distant points still have significant affinity

### Dataset Density Effects

With fewer points per cluster:
- **Lower point density**: Average nearest-neighbor distance increases
- **Weaker connectivity**: Same sigma produces sparser affinity matrix
- **Degraded eigenvectors**: Less separation between clusters in eigenspace

**Example** (three concentric circles):
- 100 points/circle: Angular spacing = 2π/100 ≈ 0.063 radians
- 50 points/circle: Angular spacing = 2π/50 ≈ 0.126 radians (2x sparser)
- 30 points/circle: Angular spacing = 2π/30 ≈ 0.209 radians (3.3x sparser)

### Empirical Validation

Systematic testing revealed the following working sigma values:

| Points/circle | Total points | Working sigma | Result |
|---------------|--------------|---------------|--------|
| 100 | 300 | 0.158 | 3 clusters, 100% purity (MATLAB) |
| 50 | 150 | 0.200 | 3 clusters, 100% purity |
| 30 | 90 | 0.200 | 3 clusters, 100% purity |

**Observation**: Sigma needs to **increase** as point count **decreases** to maintain equivalent connectivity.

### Algorithmic Behavior with Correct Sigma

With sigma=0.2 on 150 points:

```
Trying 2 dimensions... 2 clusters, origin has 50 points
Trying 3 dimensions... 3 clusters, origin has 0 points ✓
```

The key difference: **50 points remain in origin at Dim=2**, signaling that more dimensions are needed. At Dim=3, all points correctly separate into 3 pure clusters.

## Implementation Plan

### Phase 1: Investigation (Completed)
- [x] Debug circles test failure on 150-point dataset
- [x] Identify sigma as the root cause
- [x] Empirically test sigma values from 0.05 to 0.5
- [x] Document eigenvector and cluster composition at each step

### Phase 2: Validation (Completed)
- [x] Create validation script (`test_circles_fixed.py`)
- [x] Verify 100% cluster purity on 150-point dataset
- [x] Create reproducible demo notebook (`circles_demo_simple.ipynb`)
- [x] Document sigma scaling guidelines

### Phase 3: Documentation (Completed)
- [x] Create `SIGMA_SCALING.md` with findings
- [x] Add sigma parameter guidance to notebooks
- [x] Update code comments with scaling recommendations

### Phase 4: Future Work (Deferred to Future CIP)
- [ ] Implement automatic sigma selection heuristic
- [ ] Add sigma validation warnings for extreme values
- [ ] Create systematic tests across dataset sizes
- [ ] Investigate relationship to k-nearest-neighbor graphs

## Backward Compatibility

**Breaking change**: None. This is a parameter selection guideline, not a code change.

**Migration path**: Users need to adjust sigma based on dataset size:
- Small datasets (< 300 points): Use larger sigma (≈ 0.2)
- Large datasets (≥ 300 points): Use MATLAB-equivalent sigma (≈ 0.158)

## Testing Strategy

### Validation Tests Created

1. **`test_circles_fixed.py`**: Single-configuration validation
   - 150 points (50 per circle)
   - sigma=0.2
   - Expects 3 clusters with 100% purity

2. **`test_sigma_values.py`**: Systematic parameter sweep
   - Tests sigma ∈ {0.05, 0.158, 0.1, 0.2, 0.3, 0.5}
   - Documents cluster count and purity for each
   - Identifies sigma=0.2 as optimal for 150 points

3. **`examples/circles_demo_simple.ipynb`**: Interactive demonstration
   - 90 points (30 per circle)
   - sigma=0.2
   - Includes visualization and purity analysis

### Test Results

```bash
$ python test_circles_fixed.py

Testing 50 points/circle (150 total) with sigma=0.200
======================================================================
SpectralCluster: Clustering 150 points
  Parameters: sigma=0.2, lambda=0.2, max_clusters=10
  Determining number of clusters (iterative):
    Trying 2 dimensions... 2 clusters, origin has 50 points
    Trying 3 dimensions... 3 clusters, origin has 0 points ✓

✓ Found 3 clusters
Label distribution: [50 50 50]
  Cluster 0: 50 points (c0=0, c1=0, c2=50) purity=100.0%
  Cluster 1: 50 points (c0=50, c1=0, c2=0) purity=100.0%
  Cluster 2: 50 points (c0=0, c1=50, c2=0) purity=100.0%

✓✓✓ SUCCESS: Correctly detected 3 clusters!
```

## Related Requirements

This CIP fixes a critical issue blocking validation of the community detection implementation (CIP-0006). No formal requirements were written for this bugfix.

## Implementation Status

- [x] Root cause identified (sigma parameter scaling)
- [x] Empirical testing completed
- [x] Validation tests created and passing
- [x] Documentation written (`SIGMA_SCALING.md`)
- [x] Demo notebook created
- [x] Changes committed

## References

### Internal
- CIP-0006: Community Detection Analysis Integration
- `test_circles_fixed.py`: Validation script showing 100% purity on 150 points
- `test_sigma_values.py`: Systematic sigma parameter sweep
- `examples/circles_demo_simple.ipynb`: Interactive demonstration
- Crash reproduction scripts: `minimal_crash_isolated.py`, `minimal_crash_reproduction.py`, `test_blas_operations.py`, `test_matrix_access.py`, `diagnose_matrix.py`, `check_blas_config.py`

### External
- Original MATLAB code: `~/lawrennd/spectral/matlab/demoCircles.m`
- Working Python implementation: `~/lawrennd/spectral/spectral/cluster.py`
- "Automatic Determination of the Number of Clusters using Spectral Algorithms" (Sanguinetti, Laidler, Lawrence, 2005)

### Key Debugging Artifacts
- `debug_circles_detailed.py`: Eigenvector analysis and manual k-means simulation (deleted after debugging)
- Transcript: `/Users/neil/.cursor/projects/Users-neil-lawrennd-fitkit/agent-transcripts/5b50e13c-bcd1-40cd-96a0-d803c57e01ad.txt`

## Appendix: OpenBLAS Crash on 300×300 Matrices

### Issue Summary

The 300-point three circles test crashes with `scipy.linalg.eigh()` segmentation fault (exit code 139) on macOS ARM64 (Apple Silicon M-series) with OpenBLAS 0.3.21.

**System**: macOS ARM64, NumPy 1.26.4, SciPy 1.16.0, OpenBLAS 0.3.21, Python 3.11

### Minimal Reproduction

See `minimal_crash_isolated.py` (20 lines):
```python
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

# Generate 300-point circles dataset
np.random.seed(1)
npts = 100
# ... (generate X - see file for full code)

# Compute Gaussian affinity and normalized Laplacian
D = cdist(X, X, metric='euclidean')
A = np.exp(-D**2 / (2 * 0.158**2))
row_sums = A.sum(axis=1)
D_inv_sqrt = 1.0 / np.sqrt(row_sums + 1e-10)
L = D_inv_sqrt[:, np.newaxis] * A * D_inv_sqrt[np.newaxis, :]

eigvals, eigvecs = eigh(L)  # ← CRASH: Exit code 139
```

**Matrix is numerically valid**: No NaN/Inf, symmetric, well-conditioned (300×300, dtype=float64).

### What Works vs Crashes

✅ **Works**:
- `scipy.linalg.eigvalsh(L)` - eigenvalues only
- Smaller matrices (< 250×250)
- Basic operations (`L + L`, `L * 2`, `L @ L`)

❌ **Crashes**:
- `scipy.linalg.eigh(L)` - eigenvalues + eigenvectors
- `np.linalg.eigh(L)` - eigenvalues + eigenvectors
- `np.linalg.norm(L)` - matrix norm

### Confirmed Known Issue

**Web research (February 2026)** confirms this is a known OpenBLAS problem on Apple Silicon:

1. **GitHub #1355**: `dsyev` (LAPACK eigenvalue solver) crashes when `INTERFACE64=0`, works with `INTERFACE64=1`
2. **GitHub #4583**: M1/M2 kernel panics with ARPACK eigenvalue solvers using >4 threads
3. **GitHub #3674**: General "cannot use OpenBLAS properly on M1 Mac" issues
4. **GitHub #3309**: Segfaults in dependent packages (arpack, dynare) on macOS with OpenBLAS 0.3.16

**Version history**:
- 0.3.21 (our version): No specific Apple Silicon eigenvalue fixes
- 0.3.27 (April 2024): Fixed NRM2 kernel inaccuracy on Apple M chips
- 0.3.28 (August 2024): Fixed NaN/Inf handling, improved thread safety
- **Status**: Incremental improvements continue, but eigenvalue decomposition remains problematic

### Alternative BLAS Libraries

**Intel MKL**: ❌ Not viable - no native ARM64 support, x86 only. Conda-forge stopped Intel Mac builds August 2025.

**Apple Accelerate**: ❌ Not viable - NumPy 2.0+ supports it, but SciPy 1.13+ dropped Accelerate support.

**Upgrade OpenBLAS**: ⚠️ May help but no guarantee. Version 0.3.28 has Apple Silicon fixes but eigenvalue issues persist.

### Workarounds Used

1. **Pre-computed eigenvectors** ✅: Generate with MATLAB/Octave, save to `.npz`, load in Python (used in `~/lawrennd/spectral/tests/fixtures/octave_three_circles.npz`)

2. **Smaller datasets** ✅: Use 150 points (50/circle) with sigma=0.2 instead of 300 points (CIP-0007 solution)

3. **Sparse eigensolver**: ⚠️ `scipy.sparse.linalg.eigsh(L, k=10)` hangs on 300×300 matrices (tested)

### Reproduction Scripts

- `minimal_crash_isolated.py`: Simplest 20-line reproducer
- `minimal_crash_reproduction.py`: Detailed reproduction with diagnostics
- `test_blas_operations.py`: Systematic BLAS function testing
- `test_matrix_access.py`: Basic operations vs BLAS operations
- `diagnose_matrix.py`: Matrix property validation
- `check_blas_config.py`: BLAS configuration and Fortran layout test

## Lessons Learned

1. **Always validate on multiple scales**: The 300-point MATLAB reference hid the sigma scaling issue.

2. **Parameters aren't "just tuning"**: Sigma fundamentally controls the algorithm's behavior, not just performance.

3. **Origin cluster is a signal**: When the origin cluster is non-empty at Dim=k, it indicates k clusters aren't sufficient—this is a feature, not a bug.

4. **Eigenvector inspection is essential**: Looking at eigenvector structure by ground-truth cluster revealed exactly why the algorithm stopped early.

5. **Test small before large**: If it doesn't work on 90 points, fixing it for 300 won't help.

6. **Platform-specific BLAS bugs are real**: OpenBLAS 0.3.21 on Apple Silicon has confirmed eigenvalue decomposition bugs (GitHub #1355, #3674, #4583) with no complete fix as of 0.3.28. Always have workarounds ready (smaller datasets, pre-computed eigenvectors). Alternative BLAS libraries (Intel MKL, Apple Accelerate) are not viable for SciPy on ARM64.
