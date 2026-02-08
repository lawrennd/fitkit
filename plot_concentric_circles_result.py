#!/usr/bin/env python
"""Visualize concentric circles community detection result."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import sparse
import sys
sys.path.insert(0, '.')
from fitkit.community.detection import CommunityDetector

# Generate data matching MATLAB demoCircles.m
np.random.seed(1)
npts = 100
step = 2*np.pi/npts
theta = np.arange(step, 2*np.pi + step, step)
radius = np.random.randn(npts)

r1 = np.ones(npts) + 0.1*radius
r2 = 2*np.ones(npts) + 0.1*radius
r3 = 3*np.ones(npts) + 0.1*radius

circle1 = np.column_stack([r1*np.cos(theta), r1*np.sin(theta)])
circle2 = np.column_stack([r2*np.cos(theta), r2*np.sin(theta)])
circle3 = np.column_stack([r3*np.cos(theta), r3*np.sin(theta)])

X = np.vstack([circle1, circle2, circle3])
true_labels = np.array([0]*npts + [1]*npts + [2]*npts)

# Compute affinity and Laplacian
sigma = 0.05
dist_sq = cdist(X, X, metric='sqeuclidean')
A = np.exp(-dist_sq / sigma)
D = A.sum(axis=1)
D_inv_sqrt = 1.0 / np.sqrt(D)
D_inv_sqrt_mat = sparse.diags(D_inv_sqrt)
A_sparse = sparse.csr_matrix(A)
L_sparse = D_inv_sqrt_mat @ A_sparse @ D_inv_sqrt_mat

# Run detection
detector = CommunityDetector(max_communities=10, random_state=42, affinity='precomputed')
labels = detector.fit_predict(L_sparse)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# True labels
ax = axes[0]
scatter = ax.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='tab10', s=20, alpha=0.7)
ax.set_title(f'True Labels (3 Concentric Circles)', fontsize=14, fontweight='bold')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Circle (r=1,2,3)')

# Detected labels
ax = axes[1]
scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=20, alpha=0.7)
ax.set_title(f'Detected: {detector.n_communities_} communities '
             f'({detector.n_iterations_} iterations)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Detected Community')

plt.tight_layout()
plt.savefig('/Users/neil/lawrennd/fitkit/concentric_circles_detection.png', dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to: concentric_circles_detection.png")

# Print statistics
print(f"\nResults:")
print(f"  Detected {detector.n_communities_} communities in {detector.n_iterations_} iterations")
for i in range(3):
    mask = true_labels == i
    pred_in_cluster = labels[mask]
    most_common = np.bincount(pred_in_cluster).argmax()
    purity = (pred_in_cluster == most_common).sum() / len(pred_in_cluster)
    print(f"  Circle {i} (r={i+1}): {purity:.1%} purity → detected as community {most_common}")
