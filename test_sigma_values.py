#!/usr/bin/env python
"""
Test different sigma values to see which gives correct clustering.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from fitkit.community import SpectralCluster

# Generate three concentric circles (50 points per circle)
np.random.seed(1)
npts = 50
step = 2*np.pi/npts
theta = np.arange(step, 2*np.pi + step, step)
radius = np.random.randn(npts)  # Same radius noise for all three circles

r1 = np.ones(npts) + 0.1*radius
r2 = 2*np.ones(npts) + 0.1*radius
r3 = 3*np.ones(npts) + 0.1*radius

circle1 = np.column_stack([r1*np.cos(theta), r1*np.sin(theta)])
circle2 = np.column_stack([r2*np.cos(theta), r2*np.sin(theta)])
circle3 = np.column_stack([r3*np.cos(theta), r3*np.sin(theta)])

X = np.vstack([circle1, circle2, circle3])
true_labels = np.array([0]*npts + [1]*npts + [2]*npts)

print(f"Dataset shape: {X.shape}\n")

# Test different sigma values
sigma_values = [
    (0.05, "MATLAB sigma directly"),
    (np.sqrt(0.05 / 2), "sqrt(0.05/2) - converted"),
    (0.1, "0.1"),
    (0.2, "0.2"),
    (0.3, "0.3"),
    (0.5, "0.5"),
]

for sigma, desc in sigma_values:
    print(f"{'='*60}")
    print(f"Testing sigma={sigma:.4f} ({desc})")
    print(f"{'='*60}")
    
    try:
        clf = SpectralCluster(sigma=sigma, lambda_=0.2, max_clusters=10, verbose=False)
        clf.fit(X)
        
        print(f"Found {clf.n_clusters_} clusters")
        print(f"Label distribution: {np.bincount(clf.labels_)}")
        
        # Check cluster composition
        for cluster_id in range(clf.n_clusters_):
            mask = clf.labels_ == cluster_id
            if mask.sum() > 0:
                circle_0_count = mask[:npts].sum()
                circle_1_count = mask[npts:2*npts].sum()
                circle_2_count = mask[2*npts:].sum()
                purity = max(circle_0_count, circle_1_count, circle_2_count) / mask.sum()
                print(f"  Cluster {cluster_id}: {mask.sum()} points "
                      f"(c0={circle_0_count}, c1={circle_1_count}, c2={circle_2_count}) "
                      f"purity={purity:.2%}")
        
        if clf.n_clusters_ == 3:
            print("✓ SUCCESS: Found 3 clusters!")
        else:
            print(f"✗ Expected 3 clusters, found {clf.n_clusters_}")
        print()
    except Exception as e:
        print(f"ERROR: {e}\n")
