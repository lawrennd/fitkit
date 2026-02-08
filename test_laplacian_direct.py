#!/usr/bin/env python
"""Test normalized Laplacian computation directly."""
import numpy as np

# Generate simple concentric circles
np.random.seed(1)
npts = 50  # Smaller for debugging
step = 2*np.pi/npts
theta = np.arange(step, 2*np.pi + step, step)
radius = np.random.randn(npts)

r1 = np.ones(npts) + 0.1*radius
r2 = 2*np.ones(npts) + 0.1*radius

circle1 = np.column_stack([r1*np.cos(theta), r1*np.sin(theta)])
circle2 = np.column_stack([r2*np.cos(theta), r2*np.sin(theta)])
X = np.vstack([circle1, circle2])

print(f"Data: {X.shape}")

# Compute affinity (MATLAB formula exactly)
sigma2 = 0.05
n = len(X)
A = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        A[i, j] = np.exp(-np.linalg.norm(X[i] - X[j])**2 / sigma2)

print(f"Affinity matrix: {A.shape}")
print(f"  min: {A.min():.6f}, max: {A.max():.6f}")
print(f"  finite: {np.all(np.isfinite(A))}")

# Compute normalized Laplacian (MATLAB formula exactly)
D = A.sum(axis=1)
print(f"\nDegree vector D:")
print(f"  min: {D.min():.6f}, max: {D.max():.6f}")
print(f"  any zeros: {np.any(D == 0)}")
print(f"  any near-zero (<1e-10): {np.any(D < 1e-10)}")

# Check which points have zero/low degree
low_degree = np.where(D < 1e-10)[0]
if len(low_degree) > 0:
    print(f"  Points with near-zero degree: {low_degree}")
    print(f"  These points: {X[low_degree]}")
    print(f"  Their A row sums: {A[low_degree].sum(axis=1)}")

# MATLAB: L = inv(diag(sqrt(D)))*(A/diag(sqrt(D)));
# This is: L = D^{-1/2} * A * D^{-1/2}
# Add small epsilon to avoid division by zero
D_safe = D + 1e-10
D_sqrt_inv = np.diag(1.0 / np.sqrt(D_safe))
L = D_sqrt_inv @ A @ D_sqrt_inv

print(f"\nNormalized Laplacian L:")
print(f"  shape: {L.shape}")
print(f"  min: {L.min():.6f}, max: {L.max():.6f}")
print(f"  finite: {np.all(np.isfinite(L))}")
print(f"  symmetric: {np.allclose(L, L.T)}")

# Try eigendecomposition
print(f"\nEigendecomposition:")
try:
    eigvals, eigvecs = np.linalg.eigh(L)
    print(f"  ✓ Success")
    print(f"  Top 5 eigenvalues: {eigvals[-5:][::-1]}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
