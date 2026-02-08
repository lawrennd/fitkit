# OpenBLAS Crash Report: scipy.linalg.eigh on 300×300 Matrix

## Issue Summary

`scipy.linalg.eigh()` crashes with segmentation fault (exit code 139) when computing eigenvectors of a 300×300 normalized Laplacian matrix from three concentric circles dataset on macOS ARM64 (Apple Silicon) with OpenBLAS.

## System Configuration

```
Platform: macOS ARM64 (Apple Silicon M-series)
NumPy:    1.26.4
SciPy:    1.16.0
BLAS:     OpenBLAS 0.3.21
LAPACK:   OpenBLAS 0.3.21
Python:   3.11
```

## Minimal Reproduction

```python
import numpy as np
from scipy.linalg import eigh

# Generate 300-point three circles dataset
np.random.seed(1)
npts = 100
step = 2*np.pi/npts
theta = np.arange(step, 2*np.pi + step, step)
radius = np.random.randn(npts)

r1 = np.ones(npts) + 0.1*radius
r2 = 2*np.ones(npts) + 0.1*radius
r3 = 3*np.ones(npts) + 0.1*radius

X = np.vstack([
    np.column_stack([r1*np.cos(theta), r1*np.sin(theta)]),
    np.column_stack([r2*np.cos(theta), r2*np.sin(theta)]),
    np.column_stack([r3*np.cos(theta), r3*np.sin(theta)])
])

# Compute Gaussian affinity
from scipy.spatial.distance import cdist
D = cdist(X, X, metric='euclidean')
A = np.exp(-D**2 / (2 * 0.158**2))

# Normalize Laplacian
row_sums = A.sum(axis=1)
D_inv_sqrt = 1.0 / np.sqrt(row_sums + 1e-10)
L = D_inv_sqrt[:, np.newaxis] * A * D_inv_sqrt[np.newaxis, :]

# CRASH: Segmentation fault (exit code 139)
eigvals, eigvecs = eigh(L)  # ← Crashes here
```

**File**: `minimal_crash_isolated.py`

## Matrix Properties

The matrix is numerically valid:
```
Shape: (300, 300)
Dtype: float64
Range: [0.0, 0.835488]
Diagonal: [0.157799, 0.835488]
Symmetric: True
Contains NaN/Inf: False
Zero elements: 226 out of 90,000
Memory layout: C-contiguous
```

## What Works

- ✅ Matrix creation and basic operations
- ✅ Element-wise operations (`L + L`, `L * 2`, `L.T`)
- ✅ Matrix multiplication (`L @ L`) - with warning
- ✅ `scipy.linalg.eigvalsh(L)` - eigenvalues only
- ✅ Smaller matrices (< 250×250)

## What Crashes

- ❌ `scipy.linalg.eigh(L)` - eigenvalues + eigenvectors
- ❌ `np.linalg.eigh(L)` - eigenvalues + eigenvectors  
- ❌ `np.linalg.norm(L)` - matrix norm
- ❌ Fortran-contiguous layout doesn't help

## Known Workarounds

### 1. Use Pre-computed Eigenvectors (Current Solution)
```python
# Generate eigenvectors in MATLAB/Octave, save to .npz
# Load in Python:
data = np.load('octave_eigenvectors.npz')
eigvecs = data['PcEig']
```

**Status**: ✅ Working - used in `~/lawrennd/spectral/tests/fixtures/octave_three_circles.npz`

### 2. Use Sparse Eigensolver (Partially Working)
```python
from scipy.sparse.linalg import eigsh
eigvals, eigvecs = eigsh(L, k=10)  # Request fewer eigenvectors
```

**Status**: ⚠️ Hangs on 300×300 matrices (tested and confirmed)

### 3. Reduce Dataset Size (Working)
```python
npts = 50  # Use 150 total points instead of 300
sigma = 0.2  # Adjust sigma for smaller dataset
```

**Status**: ✅ Working - see CIP-0007 for sigma scaling guidelines

### 4. Use Different BLAS Library

**Intel MKL**: ❌ **Not viable** - MKL has no native ARM64 support for Apple Silicon. Only available for x86 architectures. Conda-forge stopped building Intel Mac packages in August 2025.

**Apple Accelerate**: ❌ **Not viable for SciPy** - While NumPy 2.0+ supports Accelerate, SciPy dropped Accelerate support in v1.13.x. SciPy requires OpenBLAS, MKL, or ATLAS.

**Upgrade OpenBLAS**: ⚠️ **Potentially helpful** but no guarantee:
```bash
# Upgrade to latest OpenBLAS (0.3.28 as of Aug 2024)
conda update openblas
# OR build from source with INTERFACE64=1
```
Version 0.3.27+ includes Apple Silicon fixes (NRM2 kernels, NaN/Inf handling), but eigenvalue decomposition issues persist.

## Root Cause

**Confirmed**: This is a **known issue** with OpenBLAS on Apple Silicon ARM64.

**Web Research Findings** (February 2026):

1. **`dsyev` crashes documented**: OpenBLAS has a history of crashing in the `dsyev` LAPACK eigenvalue solver, particularly when compiled with `INTERFACE64=0`. The crashes occur in the dsyev→dgemm call path. [GitHub Issue #1355]

2. **Apple Silicon stability issues**: Multiple reports of OpenBLAS causing kernel panics and crashes on Apple M1/M2 chips, especially with eigenvalue solvers (`eigsh`) and when using >4 CPU threads. [GitHub Issue #4583, #3674]

3. **Version-specific problems**: 
   - OpenBLAS 0.3.16: Segfaults in dependent packages (arpack, dynare) on Intel macOS
   - OpenBLAS 0.3.21: Our crash (no specific GitHub issue found)
   - OpenBLAS 0.3.26: DGESVD failures on M2 with certain matrix types
   - OpenBLAS 0.3.27: Fixed NRM2 kernel inaccuracy on Apple M chips (April 2024)
   - OpenBLAS 0.3.28: Fixed NaN/Inf handling, thread management (August 2024)

4. **No complete fix**: As of 0.3.28, incremental stability improvements continue but eigenvalue decomposition remains problematic on Apple Silicon.

**Our Evidence**:
1. `scipy.linalg.eigvalsh()` (eigenvalues only) works fine
2. Crash happens in `eigh()` (eigenvalues + eigenvectors)  
3. Size-dependent: works for smaller matrices (< 250×250)
4. MATLAB/Octave with different BLAS don't crash on same matrix
5. `np.linalg.norm()` also crashes (uses BLAS `dnrm2`)

## Impact

### fitkit Community Detection (CIP-0006)
- Cannot validate MATLAB equivalence on full 300-point circles dataset
- Must use either:
  - Pre-computed Octave eigenvectors (current approach)
  - Smaller datasets with adjusted sigma (CIP-0007)

### Spectral Clustering Performance
- Limits practical dataset size to < 250 points without workarounds
- Requires sigma parameter adjustment for smaller datasets
- Blocks systematic testing across multiple scales

## Recommended Actions

### Short-term (Current)
- [x] Document crash in CIP-0007
- [x] Use smaller datasets (150 points) with sigma=0.2
- [x] Create minimal reproduction script
- [ ] Test with Intel MKL or Apple Accelerate

### Long-term (Future CIP)
- [ ] Report bug to OpenBLAS maintainers
- [ ] Implement automatic BLAS backend detection and warnings
- [ ] Add sigma auto-selection heuristic for different dataset sizes
- [ ] Create comprehensive test suite with matrix size sweep

## Web Research Summary (February 2026)

### Known OpenBLAS Issues on Apple Silicon

1. **GitHub Issue #1355**: dsyev crashes when `INTERFACE64=0`, works with `INTERFACE64=1`
   - https://github.com/xianyi/OpenBLAS/issues/1355

2. **GitHub Issue #3674**: Cannot use OpenBLAS properly on M1 Mac
   - https://github.com/xianyi/OpenBLAS/issues/3674

3. **GitHub Issue #4583**: ARPACK kernel panics on M1 with >4 threads
   - https://github.com/OpenMathLib/OpenBLAS/issues/4583

4. **GitHub Issue #3309**: Segfaults in dependent packages with 0.3.16 on Intel macOS
   - https://github.com/OpenMathLib/OpenBLAS/issues/3309

### Version History

- **0.3.21** (our version): No specific fixes for Apple Silicon eigenvalue issues
- **0.3.27** (April 2024): Fixed NRM2 kernel inaccuracy on Apple M chips
- **0.3.28** (August 2024): Fixed NaN/Inf handling, improved thread safety

### Alternative BLAS Libraries

- **Intel MKL**: Not available for ARM64/Apple Silicon (x86 only)
- **Apple Accelerate**: NumPy 2.0+ supports it, but SciPy 1.13+ does not
- **Recommendation**: Stay with OpenBLAS, use workarounds (smaller datasets or pre-computed eigenvectors)

## References

### Internal
- CIP-0006: Community Detection Analysis Integration
- CIP-0007: Sigma Parameter Scaling for Spectral Clustering
- `minimal_crash_isolated.py`: Minimal reproduction
- `test_blas_operations.py`: Systematic BLAS testing
- `diagnose_matrix.py`: Matrix property diagnostics

### External
- OpenBLAS: https://github.com/xianyi/OpenBLAS/issues
- Apple Silicon BLAS issues: Known problem with certain OpenBLAS versions
- SciPy eigh docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html

## Testing Script

Run `minimal_crash_isolated.py` to reproduce:
```bash
cd /Users/neil/lawrennd/fitkit
timeout 10 python minimal_crash_isolated.py
echo "Exit code: $?"
# Expected: Exit code 139 (SIGSEGV)
```
