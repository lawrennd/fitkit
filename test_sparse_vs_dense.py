#!/usr/bin/env python3
"""
Test sparse vs dense networks.

Key insight from tests:
- Dense → λ₂^T small
- Sparse → λ₂^T larger

So maybe "nested" in the paper means SPARSE nested, not dense?
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

def generate_sparse_nested(n_countries=60, n_products=80, min_degree=2):
    """
    Generate SPARSE nested: each country/product has minimum degree.
    This creates a sparse but connected structure.
    """
    np.random.seed(42)
    M = np.zeros((n_countries, n_products))
    
    # Ensure minimum connectivity
    for c in range(n_countries):
        # Each country exports at least min_degree products
        products = np.random.choice(n_products, size=min_degree, replace=False)
        M[c, products] = 1
    
    for p in range(n_products):
        # Each product exported by at least min_degree countries
        if M[:, p].sum() < min_degree:
            countries = np.random.choice(n_countries, size=min_degree, replace=False)
            M[countries, p] = 1
    
    return M

def generate_path_graph(n=60):
    """
    Generate a PATH graph (1D structure).
    Countries: 0-1-2-3-...-59
    Each country connected only to neighbors.
    This is maximally 1D!
    """
    M = np.zeros((n, n))
    for i in range(n-1):
        M[i, i+1] = 1
        M[i+1, i] = 1  # Symmetric for undirected
    
    # For bipartite, duplicate structure
    M_bipartite = np.zeros((n, n))
    for i in range(n):
        # Each country exports to products with similar complexity
        if i > 0:
            M_bipartite[i, i-1] = 1
        if i < n-1:
            M_bipartite[i, i+1] = 1
    
    return M_bipartite

print("=" * 70)
print("TESTING SPARSE VS DENSE STRUCTURES")
print("=" * 70)

# Test 1: Very sparse random
print("\n1. VERY SPARSE RANDOM (min_degree=2)")
print("-" * 70)
M_sparse = generate_sparse_nested(min_degree=2)
eigvals_sparse = compute_eigenvalues(M_sparse)
print(f"  Density: {M_sparse.sum()/M_sparse.size:.2%}")
print(f"  λ₁^T: {eigvals_sparse[0]:.6f}")
print(f"  λ₂^T: {eigvals_sparse[1]:.6f}")
print(f"  λ₃^T: {eigvals_sparse[2]:.6f}")

# Test 2: Slightly less sparse
print("\n2. LESS SPARSE (min_degree=5)")
print("-" * 70)
M_less_sparse = generate_sparse_nested(min_degree=5)
eigvals_less_sparse = compute_eigenvalues(M_less_sparse)
print(f"  Density: {M_less_sparse.sum()/M_less_sparse.size:.2%}")
print(f"  λ₁^T: {eigvals_less_sparse[0]:.6f}")
print(f"  λ₂^T: {eigvals_less_sparse[1]:.6f}")
print(f"  λ₃^T: {eigvals_less_sparse[2]:.6f}")

# Test 3: Path-like structure (maximally 1D)
print("\n3. PATH-LIKE STRUCTURE (maximally 1D)")
print("-" * 70)
M_path = generate_path_graph(60)
eigvals_path = compute_eigenvalues(M_path)
print(f"  Density: {M_path.sum()/M_path.size:.2%}")
print(f"  λ₁^T: {eigvals_path[0]:.6f}")
print(f"  λ₂^T: {eigvals_path[1]:.6f}")
print(f"  λ₃^T: {eigvals_path[2]:.6f}")

# Test 4: Complete bipartite (for comparison)
print("\n4. COMPLETE BIPARTITE (maximally dense)")
print("-" * 70)
M_complete = np.ones((60, 80))
eigvals_complete = compute_eigenvalues(M_complete)
print(f"  Density: {M_complete.sum()/M_complete.size:.2%}")
print(f"  λ₁^T: {eigvals_complete[0]:.6f}")
print(f"  λ₂^T: {eigvals_complete[1]:.6f}")
print(f"  λ₃^T: {eigvals_complete[2]:.6f}")

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("\nCheeger's inequality connects λ₂^L to conductance Φ:")
print("  λ₂^L ≥ Φ²/2")
print()
print("  Small Φ (bottleneck) → Small λ₂^L → Large λ₂^T")
print("  Large Φ (well-connected) → Large λ₂^L → Small λ₂^T")
print()
print("So the diagnostic should be:")
print("  • λ₂^T ≈ 0 (λ₂^L ≈ 1): Well-connected, no bottleneck")
print("  • λ₂^T ≈ 1 (λ₂^L ≈ 0): Severe bottleneck (disconnected)")
print()
print("For bipartite networks:")
print("  • Dense/nested → High conductance → λ₂^T small")
print("  • Sparse communities → Low conductance → λ₂^T large???")
print()
print("This CONTRADICTS the paper's statement!")
print("Need to re-examine what the paper means by 'nested → λ₂^T ≈ 1'")
print("=" * 70)
