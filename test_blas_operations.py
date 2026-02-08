#!/usr/bin/env python
"""
Test which BLAS/LAPACK operations work on the 300x300 Laplacian.
"""
import numpy as np
import sys

print("Setting up 300x300 normalized Laplacian...")
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
print(f"✓ Laplacian ready: {L.shape}, dtype={L.dtype}\n")

def test_operation(name, func):
    """Test a BLAS/LAPACK operation."""
    print(f"Testing {name}...", end='', flush=True)
    sys.stdout.flush()
    try:
        result = func(L)
        print(f" ✓ OK (result shape: {np.array(result).shape if hasattr(result, 'shape') else type(result).__name__})")
        return True
    except Exception as e:
        print(f" ✗ {type(e).__name__}: {e}")
        return False

# Test various operations
print("="*70)
print("BASIC LINEAR ALGEBRA")
print("="*70)
test_operation("np.linalg.norm", lambda m: np.linalg.norm(m))
test_operation("np.trace", lambda m: np.trace(m))
test_operation("np.linalg.det", lambda m: np.linalg.det(m))
test_operation("np.linalg.matrix_rank", lambda m: np.linalg.matrix_rank(m))

print("\n" + "="*70)
print("EIGENVALUE OPERATIONS")
print("="*70)
test_operation("scipy.linalg.eigvalsh (eigenvalues only)", 
               lambda m: __import__('scipy.linalg').linalg.eigvalsh(m))

test_operation("scipy.linalg.eigh (eigenvalues + eigenvectors)",
               lambda m: __import__('scipy.linalg').linalg.eigh(m))

print("\n" + "="*70)
print("SPARSE EIGENVALUE OPERATIONS")
print("="*70)
test_operation("scipy.sparse.linalg.eigsh (sparse, k=10)",
               lambda m: __import__('scipy.sparse.linalg').eigsh(m, k=10))

print("\n" + "="*70)
print("ALTERNATIVE DENSE EIGENVECTOR METHODS")
print("="*70)
test_operation("np.linalg.eigh",
               lambda m: np.linalg.eigh(m))

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("If scipy.linalg.eigh crashes but eigvalsh works, it's an eigenvector")
print("computation issue in BLAS/LAPACK, not the eigenvalue solver itself.")
