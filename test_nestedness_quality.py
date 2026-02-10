#!/usr/bin/env python3
"""
Test the quality of nested network generation.

From paper (line 448): "Perfectly nested → λ₂^T ≈ 1 (equivalently λ₂^L ≈ 0)"

So a good nested network should have:
- λ₂^T close to 1 (high persistence, no bottleneck)
- λ₂^L close to 0 (small Laplacian gap, high conductance)
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

def compute_eigenvalues(M):
    """Compute eigenvalues of transition matrix."""
    k_c = M.sum(axis=1) + 1e-10
    k_p = M.sum(axis=0) + 1e-10
    D_c_inv = sparse.diags(1.0 / k_c)
    D_p_inv = sparse.diags(1.0 / k_p)
    M_sp = sparse.csr_matrix(M)
    T = D_c_inv @ M_sp @ D_p_inv @ M_sp.T
    
    n_countries = M.shape[0]
    eigvals_T, _ = eigsh(T, k=min(10, n_countries-2), which='LA')
    eigvals_T = np.sort(eigvals_T)[::-1]
    eigvals_L = 1.0 - eigvals_T
    
    return eigvals_T, eigvals_L

def measure_nestedness(M):
    """Measure actual nestedness of matrix."""
    n_countries, n_products = M.shape
    
    # Sort countries by diversification (degree)
    country_degrees = M.sum(axis=1)
    country_order = np.argsort(country_degrees)[::-1]  # Most diverse first
    
    # Sort products by ubiquity (degree)
    product_degrees = M.sum(axis=0)
    product_order = np.argsort(product_degrees)[::-1]  # Most ubiquitous first
    
    # Reorder matrix
    M_sorted = M[country_order, :][:, product_order]
    
    # Count violations of perfect nestedness
    # In perfect nesting: if M[i,j]=1, then M[i',j]=1 for all i' > i
    # and M[i,j']=1 for all j' > j
    violations = 0
    total_ones = M.sum()
    
    for i in range(n_countries):
        for j in range(n_products):
            if M_sorted[i, j] == 1:
                # Check if any less capable country (i'>i) lacks this product
                violations += np.sum(M_sorted[i+1:, j] == 0)
    
    nestedness_score = 1.0 - (violations / (total_ones * n_countries))
    return nestedness_score, M_sorted

def generate_nested_network(n_countries=60, n_products=80, noise=0.05, seed=42):
    """Current nested network generator."""
    np.random.seed(seed)
    capability = np.sort(np.random.uniform(0, 1, n_countries))
    complexity = np.sort(np.random.uniform(0, 1, n_products))
    
    M = np.zeros((n_countries, n_products))
    for c in range(n_countries):
        for p in range(n_products):
            if capability[c] >= complexity[p]:
                if np.random.random() > noise:
                    M[c, p] = 1
            else:
                if np.random.random() < noise:
                    M[c, p] = 1
    return M

# Test current generator
print("="*70)
print("TESTING NESTEDNESS QUALITY")
print("="*70)

for noise in [0.01, 0.05, 0.10]:
    print(f"\nNoise = {noise}")
    print("-"*70)
    M = generate_nested_network(n_countries=60, n_products=80, noise=noise, seed=42)
    eigvals_T, eigvals_L = compute_eigenvalues(M)
    nestedness, _ = measure_nestedness(M)
    
    print(f"  Density: {M.sum()/M.size:.2%}")
    print(f"  Nestedness score: {nestedness:.4f}")
    print(f"  λ₂^T: {eigvals_T[1]:.4f}  (target: ≈1.0 for perfect nesting)")
    print(f"  λ₂^L: {eigvals_L[1]:.4f}  (target: ≈0.0 for perfect nesting)")
    print(f"  Cheeger bound on conductance: Φ ≥ √(2λ₂^L) = {np.sqrt(2*eigvals_L[1]):.4f}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print("Problem: Even with 1% noise, λ₂^T = 0.21 (far from 1)")
print("         Expected: λ₂^T ≈ 1 for truly nested network")
print("\nPossible causes:")
print("  1. Nested structure too 'loose' (wide capability/complexity ranges)")
print("  2. Sparse matrix (many gaps in the nesting)")
print("  3. Generator not creating tight triangular structure")
print("\nNext: Check if a PERFECT nested matrix gives λ₂^T ≈ 1")
print("="*70)
