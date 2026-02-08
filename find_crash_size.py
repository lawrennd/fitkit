#!/usr/bin/env python
"""Find at what size the computation crashes."""
import numpy as np
from scipy.spatial.distance import cdist

np.random.seed(1)

for n_per_circle in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    npts = n_per_circle
    npts_total = 3 * npts
    
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
    
    print(f"Testing n_per_circle={npts}, total={npts_total}...", end=" ", flush=True)
    
    try:
        sigma = 0.05
        dist_sq = cdist(x, x, metric='sqeuclidean')
        A = np.exp(-dist_sq / sigma)
        D = A.sum(axis=1)
        D_sqrt_inv = 1.0 / np.sqrt(D)
        
        # Compute L = D^{-1/2} A D^{-1/2} element-wise
        # D_sqrt_inv[i] = 1/sqrt(D[i]), so L[i,j] = A[i,j] * D_sqrt_inv[i] * D_sqrt_inv[j]
        L = np.zeros((npts_total, npts_total))
        for i in range(npts_total):
            for j in range(npts_total):
                L[i, j] = A[i, j] * D_sqrt_inv[i] * D_sqrt_inv[j]
       
        eigvals, eigvecs = np.linalg.eigh(L)
        print(f"✓ OK (top eigval: {eigvals[-1]:.6f})")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        break
