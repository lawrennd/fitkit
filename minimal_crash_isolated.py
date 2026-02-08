#!/usr/bin/env python
"""
Minimal isolated test - find exact hanging point.
"""
import numpy as np

print("Generating 300-point circles...")
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
print(f"✓ Data: {X.shape}")

print("\nComputing affinity...")
from scipy.spatial.distance import cdist
D = cdist(X, X, metric='euclidean')
A = np.exp(-D**2 / (2 * 0.158**2))
print(f"✓ Affinity: {A.shape}")

print("\nComputing normalized Laplacian...")
row_sums = A.sum(axis=1)
D_inv_sqrt = 1.0 / np.sqrt(row_sums + 1e-10)
L = D_inv_sqrt[:, np.newaxis] * A * D_inv_sqrt[np.newaxis, :]
print(f"✓ Laplacian: {L.shape}, symmetric={np.allclose(L, L.T)}")

print("\nAttempting scipy.linalg.eigh...")
import sys
sys.stdout.flush()

from scipy.linalg import eigh
eigvals, eigvecs = eigh(L)

print(f"✓ SUCCESS: {eigvecs.shape}")
