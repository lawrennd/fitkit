#!/usr/bin/env python
"""Test circles detection with new implementation."""
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from fitkit.community import CommunityDetector

# Generate concentric circles
np.random.seed(1)
npts = 100
step = 2*np.pi/npts
theta = np.arange(step, 2*np.pi + step, step)
radius = np.random.randn(npts)

r1 = np.ones(npts) + 0.1*radius
r2 = 2*np.ones(npts) + 0.1*np.random.randn(npts)
r3 = 3*np.ones(npts) + 0.1*np.random.randn(npts)

circle1 = np.column_stack([r1*np.cos(theta), r1*np.sin(theta)])
circle2 = np.column_stack([r2*np.cos(theta), r2*np.sin(theta)])
circle3 = np.column_stack([r3*np.cos(theta), r3*np.sin(theta)])

X = np.vstack([circle1, circle2, circle3])
print(f"Data shape: {X.shape}")

# Compute Gaussian affinity matrix
print("Computing affinity matrix...")
dist_sq = cdist(X, X, metric='sqeuclidean')
A = np.exp(-dist_sq / 0.05)

# Convert to normalized Laplacian
D = A.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-10))
L = D_inv_sqrt @ A @ D_inv_sqrt

print("Running community detection...")
import sys
sys.stdout.flush()

detector = CommunityDetector(max_communities=5, random_state=42,  # Lower max for speed
                             affinity='precomputed')
print("Created detector, calling fit_predict...")
sys.stdout.flush()
labels = detector.fit_predict(L)
print("fit_predict returned")
sys.stdout.flush()

print(f"\nDetected {detector.n_communities_} communities")
print(f"Label distribution: {np.bincount(labels)}")

# Check if we found 3 clusters
if detector.n_communities_ == 3:
    print("✓ SUCCESS: Found 3 communities as expected!")
else:
    print(f"✗ FAILURE: Expected 3 communities, found {detector.n_communities_}")
