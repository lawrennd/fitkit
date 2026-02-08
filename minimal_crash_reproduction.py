#!/usr/bin/env python
"""
Minimal reproduction of scipy.linalg.eigh crash on 300x300 normalized Laplacian.

This script reproduces the BLAS segfault that occurs when computing eigenvectors
of a 300x300 normalized Laplacian matrix from three concentric circles.
"""
import numpy as np
from scipy.linalg import eigh

print("Generating three concentric circles (300 points)...")
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
print(f"Data shape: {X.shape}")

print("\nComputing Gaussian affinity matrix...")
sigma = 0.158  # MATLAB-equivalent sigma
from scipy.spatial.distance import cdist
D = cdist(X, X, metric='euclidean')
A = np.exp(-D**2 / (2 * sigma**2))
print(f"Affinity matrix shape: {A.shape}")
print(f"Affinity matrix dtype: {A.dtype}")
print(f"Affinity range: [{A.min():.6f}, {A.max():.6f}]")

print("\nComputing normalized Laplacian...")
row_sums = A.sum(axis=1)
D_inv_sqrt = 1.0 / np.sqrt(row_sums + 1e-10)
L = D_inv_sqrt[:, np.newaxis] * A * D_inv_sqrt[np.newaxis, :]
print(f"Laplacian shape: {L.shape}")
print(f"Laplacian dtype: {L.dtype}")
print(f"Laplacian range: [{L.min():.6f}, {L.max():.6f}]")
print(f"Laplacian is symmetric: {np.allclose(L, L.T)}")

# Check for problematic values
print(f"\nMatrix diagnostics:")
print(f"  Contains NaN: {np.any(np.isnan(L))}")
print(f"  Contains Inf: {np.any(np.isinf(L))}")
print(f"  Condition number: {np.linalg.cond(L):.2e}")

print("\n" + "="*70)
print("Attempting to compute eigenvectors with scipy.linalg.eigh...")
print("(This will crash with BLAS segfault or hang indefinitely)")
print("="*70)

try:
    eigvals, eigvecs = eigh(L)
    print(f"\n✓ SUCCESS (unexpected!): Computed {eigvecs.shape[1]} eigenvectors")
except Exception as e:
    print(f"\n✗ FAILED with exception: {type(e).__name__}: {e}")

print("\nIf you see this line, the computation completed without crashing.")
