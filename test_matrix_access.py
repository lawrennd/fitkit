#!/usr/bin/env python
"""
Test basic matrix access and operations.
"""
import numpy as np
import sys

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
row_sums = A.sum(axis=1)
D_inv_sqrt = 1.0 / np.sqrt(row_sums + 1e-10)
L = D_inv_sqrt[:, np.newaxis] * A * D_inv_sqrt[np.newaxis, :]

print(f"✓ Matrix created: {L.shape}, dtype={L.dtype}")
print(f"  Memory: {L.nbytes / 1024:.1f} KB")
print(f"  C-contiguous: {L.flags.c_contiguous}")
print(f"  F-contiguous: {L.flags.f_contiguous}")
print(f"  Owndata: {L.flags.owndata}")

print("\nTesting basic operations:")
print(f"  L[0,0] = {L[0,0]:.6f}")
print(f"  L.shape = {L.shape}")
print(f"  L.min() = {L.min():.6f}")
print(f"  L.max() = {L.max():.6f}")
print(f"  L.sum() = {L.sum():.2f}")

print("\nTesting numpy operations (no BLAS):")
print("  L + L...", end='', flush=True); result = L + L; print(" ✓")
print("  L * 2...", end='', flush=True); result = L * 2; print(" ✓")
print("  L.T...", end='', flush=True); result = L.T; print(" ✓")

print("\nTesting BLAS operations:")
print("  L @ L (matrix multiply)...", end='', flush=True)
sys.stdout.flush()
result = L @ L
print(" ✓")

print("  np.linalg.norm(L)...", end='', flush=True)
sys.stdout.flush()
result = np.linalg.norm(L)
print(f" ✓ {result:.2f}")

print("\n✓ All tests passed!")
