#!/usr/bin/env python
"""Replicate MATLAB demoCircles computation EXACTLY."""
import numpy as np

# Match MATLAB: randn('seed',1); rand('seed',1);
np.random.seed(1)

npts = 20  # Smaller for testing
step = 2*np.pi/npts
theta = np.arange(step, 2*np.pi + step, step)
radius = np.random.randn(npts)

# MATLAB: r1 = [ones(1,npts)+0.1*radius];
r1 = np.ones(npts) + 0.1*radius
r2 = 2*np.ones(npts) + 0.1*radius
r3 = 3*np.ones(npts) + 0.1*radius

# MATLAB: x = [(r1.*cos(theta))', (r1.*sin(theta))'; ...]
circle1 = np.column_stack([r1*np.cos(theta), r1*np.sin(theta)])
circle2 = np.column_stack([r2*np.cos(theta), r2*np.sin(theta)])
circle3 = np.column_stack([r3*np.cos(theta), r3*np.sin(theta)])

x = np.vstack([circle1, circle2, circle3])
print(f"x: {x.shape}")
print(f"Sample points: {x[0]}, {x[npts]}, {x[2*npts]}")

# MATLAB: sigma2 = 0.05 (but used as sigma in formula)
sigma = 0.05

# MATLAB: A=zeros(npts,npts); (but npts is now 3*npts)
npts_total = 3 * npts

print(f"\nComputing A ({npts_total}x{npts_total}) using vectorized approach...")
# Compute pairwise distances using broadcasting
from scipy.spatial.distance import cdist
dist_sq = cdist(x, x, metric='sqeuclidean')
A = np.exp(-dist_sq / sigma)

print(f"A computed:")
print(f"  A[0,0] = {A[0,0]} (should be 1.0)")
print(f"  A[0,1] = {A[0,1]}")
print(f"  min = {A.min()}, max = {A.max()}")

# MATLAB: D = (sum(A,2));
D = A.sum(axis=1)
print(f"\nD = sum(A,2):")
print(f"  shape: {D.shape}")
print(f"  min = {D.min()}, max = {D.max()}")
print(f"  any zeros: {np.any(D == 0)}")

# MATLAB: L = inv(diag(sqrt(D)))*(A/diag(sqrt(D)));
# In MATLAB, A/B means A*inv(B) (right division)
# So: L = D^{-1/2} * A * D^{-1/2}
D_sqrt = np.sqrt(D)
D_sqrt_inv = 1.0 / D_sqrt

print(f"\n1/sqrt(D):")
print(f"  min = {D_sqrt_inv.min()}, max = {D_sqrt_inv.max()}")
print(f"  finite: {np.all(np.isfinite(D_sqrt_inv))}")

# Create diagonal matrices
D_sqrt_inv_mat = np.diag(D_sqrt_inv)
print(f"\nDiag matrix: {D_sqrt_inv_mat.shape}")

# Compute L using explicit loop to avoid matmul issues
print(f"\nComputing L row by row...")
L = np.zeros((npts_total, npts_total))
for i in range(npts_total):
    for j in range(npts_total):
        L[i, j] = A[i, j] / (D_sqrt[i] * D_sqrt[j])

print(f"L computed:")
print(f"  shape: {L.shape}")
print(f"  symmetric: {np.allclose(L, L.T)}")
print(f"  finite: {np.all(np.isfinite(L))}")
print(f"  min = {L.min()}, max = {L.max()}")

# Try eigendecomposition
print(f"\nEigendecomposition...")
try:
    eigvals, eigvecs = np.linalg.eigh(L)
    print(f"  ✓ SUCCESS!")
    print(f"  Top 5 eigenvalues: {eigvals[-5:][::-1]}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
