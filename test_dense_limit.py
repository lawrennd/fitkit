#!/usr/bin/env python3
"""
Test the dense limit: Does high density → λ₂^T ≈ 1?

Hypothesis: The paper's "perfectly nested → λ₂^T ≈ 1" might mean
the dense limit where almost all entries are 1.
"""
import numpy as np
from scipy import linalg

def build_transition_matrix(M):
    """Build T = D_c^{-1} M D_p^{-1} M^T."""
    D_c = M.sum(axis=1) + 1e-10
    D_p = M.sum(axis=0) + 1e-10
    
    M_weighted = M / D_p[np.newaxis, :]
    MM_T = M @ M_weighted.T
    T = MM_T / D_c[:, np.newaxis]
    
    return T

def compute_eigenvalues(M):
    """Compute eigenvalues of T."""
    T = build_transition_matrix(M)
    eigvals_T, _ = linalg.eig(T)
    eigvals_T = np.real(eigvals_T)
    eigvals_T = np.sort(eigvals_T)[::-1]
    return eigvals_T

print("=" * 70)
print("TESTING DENSITY VS λ₂^T")
print("=" * 70)
print("\nHypothesis: High density → λ₂^T ≈ 1")

for density in [0.5, 0.7, 0.9, 0.95, 0.99, 1.0]:
    print(f"\nDensity = {density:.0%}")
    print("-" * 70)
    
    if density < 1.0:
        np.random.seed(42)
        M = (np.random.random((60, 80)) < density).astype(float)
    else:
        M = np.ones((60, 80))
    
    eigvals = compute_eigenvalues(M)
    print(f"  λ₁^T: {eigvals[0]:.6f}")
    print(f"  λ₂^T: {eigvals[1]:.6f}")
    print(f"  λ₃^T: {eigvals[2]:.6f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("\nDid density=100% give λ₂^T ≈ 1?")
print("If yes: The paper's 'nested' means dense/complete bipartite")
print("If no: Something else is going on with the theory")
print("=" * 70)
