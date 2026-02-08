#!/usr/bin/env python
"""
Diagnose numerical issues in the 300x300 Laplacian.
"""
import numpy as np

print("Creating 300x300 Laplacian...")
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

print("\nAffinity matrix diagnostics:")
print(f"  Shape: {A.shape}")
print(f"  Range: [{A.min():.10f}, {A.max():.10f}]")
print(f"  Diagonal: [{A.diagonal().min():.10f}, {A.diagonal().max():.10f}]")
print(f"  Row sums: [{A.sum(axis=1).min():.6f}, {A.sum(axis=1).max():.6f}]")
print(f"  Contains zeros: {np.any(A == 0)}")
print(f"  Contains values < 1e-100: {np.any(A < 1e-100)}")

row_sums = A.sum(axis=1)
print(f"\nRow sum statistics:")
print(f"  Min: {row_sums.min():.10f}")
print(f"  Max: {row_sums.max():.10f}")
print(f"  Contains zeros: {np.any(row_sums == 0)}")
print(f"  Contains values < 1e-10: {np.any(row_sums < 1e-10)}")

D_inv_sqrt = 1.0 / np.sqrt(row_sums + 1e-10)
print(f"\nD^{{-1/2}} statistics:")
print(f"  Range: [{D_inv_sqrt.min():.10f}, {D_inv_sqrt.max():.10f}]")
print(f"  Contains inf: {np.any(np.isinf(D_inv_sqrt))}")
print(f"  Contains nan: {np.any(np.isnan(D_inv_sqrt))}")

L = D_inv_sqrt[:, np.newaxis] * A * D_inv_sqrt[np.newaxis, :]

print(f"\nNormalized Laplacian diagnostics:")
print(f"  Shape: {L.shape}")
print(f"  Range: [{L.min():.10f}, {L.max():.10f}]")
print(f"  Diagonal range: [{L.diagonal().min():.10f}, {L.diagonal().max():.10f}]")
print(f"  Contains inf: {np.any(np.isinf(L))}")
print(f"  Contains nan: {np.any(np.isnan(L))}")
print(f"  Contains zeros: {np.sum(L == 0)} elements")
print(f"  Symmetric: {np.allclose(L, L.T)}")
print(f"  Positive semi-definite diagonal: {np.all(L.diagonal() >= -1e-10)}")

# Check eigenvalue bounds
print(f"\nChecking matrix properties:")
print(f"  Trace: {np.trace(L):.6f}")
print(f"  Row sums: [{L.sum(axis=1).min():.6f}, {L.sum(axis=1).max():.6f}]")

# Try to identify problematic rows/cols
row_norms = np.sqrt(np.sum(L**2, axis=1))
print(f"  Row L2 norms: [{row_norms.min():.6f}, {row_norms.max():.6f}]")
problematic_rows = np.where(row_norms < 1e-10)[0]
if len(problematic_rows) > 0:
    print(f"  ⚠ Found {len(problematic_rows)} near-zero rows: {problematic_rows[:10]}")
