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
```bash
# Install Intel MKL or Apple Accelerate
conda install numpy="*=*_mkl_*"
# OR
pip install numpy-mkl
```

**Status**: ⚠️ Untested - may resolve issue

## Root Cause

**Hypothesis**: OpenBLAS 0.3.21 on Apple Silicon ARM64 has a bug in the LAPACK `dsyev`/`dsyevd` routines used by `scipy.linalg.eigh()` for dense symmetric eigenvalue problems. The crash occurs specifically when computing eigenvectors (not eigenvalues) of certain 300×300 matrices.

**Evidence**:
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
