#!/usr/bin/env python
"""Test using spectral package directly."""
import numpy as np
import sys
sys.path.insert(0, '/Users/neil/lawrennd/spectral')

from spectral import SpectralCluster

# Generate concentric circles (matching MATLAB)
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

# MATLAB uses sigma=0.05, which in their formula is exp(-d^2/sigma)
# Python SpectralCluster uses exp(-d^2/(2*sigma^2))
# So: sigma_python = sqrt(sigma_matlab / 2) = sqrt(0.05/2) ≈ 0.158
sigma = np.sqrt(0.05 / 2)
print(f"Using sigma={sigma:.4f}")

print("Running SpectralCluster...")
clf = SpectralCluster(sigma=sigma, max_clusters=10, verbose=True)
clf.fit(X)

print(f"\n✓ Found {clf.n_clusters_} clusters")
print(f"Label distribution: {np.bincount(clf.labels_)}")
