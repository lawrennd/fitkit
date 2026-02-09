---
id: "2026-02-09_openblas-crash-resolution"
title: "Investigate and resolve OpenBLAS eigenvalue decomposition crash on Apple Silicon"
status: "Proposed"
priority: "Medium"
created: "2026-02-09"
last_updated: "2026-02-09"
category: "infrastructure"
related_cips: ["0007"]
owner: "Unassigned"
dependencies: []
tags:
- backlog
- infrastructure
- blas
- apple-silicon
- eigenvalue
- crash
---

# Task: Resolve OpenBLAS Crash on Apple Silicon

> **Note**: This task tracks resolution of the OpenBLAS 0.3.21 crash on macOS ARM64 that blocks 300-point spectral clustering validation.

## Description

`scipy.linalg.eigh()` crashes with segmentation fault (exit code 139) when computing eigenvectors of 300×300 normalized Laplacian matrices on Apple Silicon M-series chips with OpenBLAS 0.3.21. This is a **confirmed known issue** (GitHub #1355, #3674, #4583) affecting the community detection implementation.

**Current impact**:
- Cannot validate MATLAB equivalence on full 300-point circles dataset
- Must use workarounds: smaller datasets (150 points, sigma=0.2) or pre-computed Octave eigenvectors
- Limits systematic testing across multiple scales

**Root cause**: OpenBLAS bug in LAPACK `dsyev`/`dsyevd` routines (eigenvalue decomposition) on Apple Silicon. The crash occurs specifically when computing eigenvectors (eigenvalues-only works fine).

## Acceptance Criteria

- [ ] Identify a stable BLAS/LAPACK configuration that doesn't crash on 300×300 eigenvalue decomposition
- [ ] Validate that `scipy.linalg.eigh()` works on the problematic 300×300 Laplacian matrix
- [ ] Run full 300-point circles test with sigma=0.158 (MATLAB equivalent) and achieve 3 clusters with 100% purity
- [ ] Document the solution and update installation instructions if needed
- [ ] Update CIP-0007 to reflect resolution

## Implementation Notes

### Investigation Steps

1. **Test OpenBLAS upgrade**: Try latest version (0.3.28 as of Aug 2024)
   ```bash
   conda update openblas
   python minimal_crash_isolated.py  # Test if crash persists
   ```

2. **Try building OpenBLAS with INTERFACE64=1**: GitHub #1355 reports this fixes `dsyev` crashes
   ```bash
   # Build from source with 8-byte integer interface
   git clone https://github.com/OpenMathLib/OpenBLAS.git
   cd OpenBLAS
   make INTERFACE64=1 TARGET=ARMV8
   # Link NumPy/SciPy against custom build
   ```

3. **Test alternative NumPy build**: Try conda-forge osx-arm64 packages with different OpenBLAS versions
   ```bash
   conda install -c conda-forge "numpy>=2.0" "scipy>=1.16"
   ```

4. **Investigate thread count**: GitHub #4583 reports crashes with >4 threads on M1
   ```bash
   export OMP_NUM_THREADS=1
   python minimal_crash_isolated.py
   ```

5. **Last resort - Rosetta 2**: Run x86_64 Python under Rosetta (slower but may work)
   ```bash
   arch -x86_64 /usr/bin/python3 minimal_crash_isolated.py
   ```

### Non-Viable Alternatives (Already Investigated)

- ❌ **Intel MKL**: No ARM64 support, x86 only
- ❌ **Apple Accelerate**: NumPy 2.0+ supports it, but SciPy 1.13+ dropped support
- ⚠️ **Sparse eigensolver**: `scipy.sparse.linalg.eigsh()` hangs on 300×300 matrices

### Testing

Run full validation suite after fix:
```bash
python minimal_crash_isolated.py  # Should complete without crash
python test_circles_fixed.py      # With npts=100 (300 points total)
pytest tests/test_community_detection.py -k circles
```

## Related

- CIP: 0007 (Sigma Parameter Scaling)
- CIP: 0006 (Community Detection Analysis Integration)
- Reproduction scripts: `minimal_crash_isolated.py`, `minimal_crash_reproduction.py`, `test_blas_operations.py`, `diagnose_matrix.py`, `check_blas_config.py`
- Reference: CIP-0007 Appendix (full crash documentation)
- Upstream issues: OpenBLAS GitHub #1355, #3674, #4583

## Progress Updates

### 2026-02-09

Task created. Crash documented in CIP-0007 Appendix with minimal reproduction and web research findings. Current workarounds (smaller datasets, pre-computed eigenvectors) are functional but limit testing capabilities.
