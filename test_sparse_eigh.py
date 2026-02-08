#!/usr/bin/env python
"""Test using sparse eigenvalue solver."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse as sp

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

x = np.vstack([circle1, circle2, circle3])
npts_total = 300

print(f"Data: {x.shape}")

sigma = 0.05
print(f"\nComputing affinity (vectorized)...")
dist_sq = cdist(x, x, metric='sqeuclidean')
A = np.exp(-dist_sq / sigma)
print(f"  ✓ A: {A.shape}, min={A.min():.6f}, max={A.max():.6f}")

D = A.sum(axis=1)
print(f"  ✓ D: min={D.min():.6f}, max={D.max():.6f}")

# Convert to sparse matrix
A_sparse = sp.csr_matrix(A)
print(f"  Sparse: {A_sparse.nnz}/{npts_total**2} nonzeros ({100*A_sparse.nnz/(npts_total**2):.1f}%)")

# Compute normalized Laplacian as sparse operations
D_sqrt_inv = 1.0 / np.sqrt(D)
D_sqrt_inv_mat = sp.diags(D_sqrt_inv)
L_sparse = D_sqrt_inv_mat @ A_sparse @ D_sqrt_inv_mat

print(f"\nNormalized Laplacian (sparse):")
print(f"  shape: {L_sparse.shape}")
print(f"  nnz: {L_sparse.nnz}")

# Use sparse eigensolver
from scipy.sparse.linalg import eigsh
print(f"\nUsing sparse eigensolver (eigsh) for k=10 largest eigenvalues...")
try:
    eigvals, eigvecs = eigsh(L_sparse, k=10, which='LA')
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    print(f"  ✓ SUCCESS!")
    print(f"  Top 10 eigenvalues: {eigvals}")
    
    # Now test with the community detector
    print(f"\nRunning community detection on eigenvectors...")
    import sys
    sys.path.insert(0, '.')
    from fitkit.community.detection import CommunityDetector
    
    detector = CommunityDetector(max_communities=10, random_state=42)
    labels = detector.fit_predict(None, eigenvectors=eigvecs)
    
    print(f"  Detected: {detector.n_communities_} communities")
    print(f"  Iterations: {detector.n_iterations_}")
    
    true_labels = np.array([0]*npts + [1]*npts + [2]*npts)
    for i in range(3):
        mask = true_labels == i
        pred_in_cluster = labels[mask]
        most_common = np.bincount(pred_in_cluster).argmax()
        purity = (pred_in_cluster == most_common).sum() / len(pred_in_cluster)
        print(f"  Circle {i} (r={i+1}): {purity:.1%} purity")
    
    if detector.n_communities_ == 3:
        print(f"\n✓ SUCCESS: 3 concentric circles detected!")
    else:
        print(f"\n✗ Expected 3, got {detector.n_communities_}")
        
except Exception as e:
    print(f"  ✗ FAILED: {e}")
