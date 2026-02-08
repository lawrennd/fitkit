#!/usr/bin/env python
"""
Check BLAS/LAPACK configuration and test workarounds.
"""
import numpy as np
import scipy

print("="*70)
print("BLAS/LAPACK CONFIGURATION")
print("="*70)
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")

# Check BLAS configuration
print("\nNumPy BLAS info:")
try:
    config = np.__config__.show()
    print(config)
except:
    print("  (not available)")

print("\nBLAS/LAPACK libraries in use:")
try:
    import numpy.distutils.system_info as sysinfo
    blas_info = sysinfo.get_info('blas_opt')
    lapack_info = sysinfo.get_info('lapack_opt')
    print(f"  BLAS: {blas_info.get('libraries', 'unknown')}")
    print(f"  LAPACK: {lapack_info.get('libraries', 'unknown')}")
except:
    print("  (unable to determine)")

print("\n" + "="*70)
print("CREATING TEST MATRIX")
print("="*70)

# Create the problematic matrix
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

from scipy.spatial.distance import cdist
D = cdist(X, X, metric='euclidean')
A = np.exp(-D**2 / (2 * 0.158**2))
row_sums = A.sum(axis=1)
D_inv_sqrt = 1.0 / np.sqrt(row_sums + 1e-10)
L = D_inv_sqrt[:, np.newaxis] * A * D_inv_sqrt[np.newaxis, :]

print(f"✓ Matrix ready: {L.shape}")

print("\n" + "="*70)
print("TESTING WORKAROUNDS")
print("="*70)

# Test 1: Ensure Fortran-contiguous (LAPACK prefers this)
print("\n1. Testing with Fortran-contiguous layout...")
L_fortran = np.asfortranarray(L)
print(f"   F-contiguous: {L_fortran.flags.f_contiguous}")
import sys
sys.stdout.flush()

try:
    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(L_fortran)
    print(f"   ✓ SUCCESS with Fortran layout!")
except:
    print(f"   ✗ Still crashes")

print("\nIf you see this, Fortran layout fixed it!")
