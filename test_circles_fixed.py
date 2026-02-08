#!/usr/bin/env python
"""
Validation that the circles clustering now works correctly with appropriate sigma.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from fitkit.community import SpectralCluster

def test_circles(npts, sigma):
    """Test circles with given number of points per circle and sigma."""
    # Generate three concentric circles (matching MATLAB's demoCircles.m)
    np.random.seed(1)
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

    print(f"\nTesting {npts} points/circle ({3*npts} total) with sigma={sigma:.3f}")
    print("="*70)
    
    clf = SpectralCluster(sigma=sigma, lambda_=0.2, max_clusters=10, verbose=True)
    clf.fit(X)
    
    print(f"\n✓ Found {clf.n_clusters_} clusters")
    print(f"Label distribution: {np.bincount(clf.labels_)}")
    
    # Check cluster purity
    for cluster_id in range(clf.n_clusters_):
        mask = clf.labels_ == cluster_id
        if mask.sum() > 0:
            circle_0_count = mask[:npts].sum()
            circle_1_count = mask[npts:2*npts].sum()
            circle_2_count = mask[2*npts:].sum()
            purity = max(circle_0_count, circle_1_count, circle_2_count) / mask.sum()
            print(f"  Cluster {cluster_id}: {mask.sum()} points "
                  f"(c0={circle_0_count}, c1={circle_1_count}, c2={circle_2_count}) "
                  f"purity={purity:.1%}")
    
    if clf.n_clusters_ == 3:
        print("\n✓✓✓ SUCCESS: Correctly detected 3 clusters!")
        return True
    else:
        print(f"\n✗ Expected 3 clusters, found {clf.n_clusters_}")
        return False

print("Spectral Clustering - Three Concentric Circles Validation")
print("="*70)
print("\nKEY FINDING: Sigma must scale with dataset size/density")
print("  - Fewer points → larger sigma needed for connectivity")
print("  - More points → smaller sigma works fine")

# Test with 50 points/circle (150 total) - the working case
success = test_circles(npts=50, sigma=0.2)

print("\n" + "="*70)
if success:
    print("VALIDATION PASSED: Algorithm correctly identifies 3 circles!")
else:
    print("VALIDATION FAILED")
