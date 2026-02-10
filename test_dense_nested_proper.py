#!/usr/bin/env python3
"""
Test DENSE nested structure for proper λ₂^T ≈ 1.

Hypothesis: My sparse triangular nested matrices create artificial bottlenecks
in the projected T = D_c^{-1} M D_p^{-1} M^T space.

A TRUE nested structure should have:
- Dense connections within capability bands
- Smooth transitions between bands
- High country-country similarity for similar capabilities
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
    return eigvals_T, T

def generate_dense_nested_smooth(n_countries=60, n_products=80, density=0.8):
    """
    Generate DENSE nested with smooth capability gradient.
    
    Key idea: Countries at similar capability levels export MANY of the same products.
    This creates high similarity in the projected T space.
    """
    np.random.seed(42)
    
    # Smooth capability gradient
    capability = np.linspace(0, 1, n_countries)
    complexity = np.linspace(0, 1, n_products)
    
    M = np.zeros((n_countries, n_products))
    
    for c in range(n_countries):
        for p in range(n_products):
            # Base probability from capability - complexity gap
            gap = capability[c] - complexity[p]
            base_prob = 1.0 / (1.0 + np.exp(-10 * gap))
            
            # Boost to achieve target density
            # Add extra connections for high-capability countries
            boost = capability[c] * (1 - density) + density
            prob = base_prob * boost
            
            M[c, p] = 1 if np.random.random() < prob else 0
    
    return M

def generate_very_dense_nested(n_countries=60, n_products=80, density=0.9):
    """
    Generate VERY DENSE nested structure.
    Almost all high-capability countries export almost all products.
    """
    np.random.seed(42)
    capability = np.linspace(0, 1, n_countries)
    
    M = np.zeros((n_countries, n_products))
    
    for c in range(n_countries):
        # High-capability countries export density% of all products
        # Low-capability countries export fewer
        n_exports = int(capability[c] * n_products * density + (1-density) * n_products * 0.1)
        n_exports = min(n_exports, n_products)
        
        # Export the simplest n_exports products
        M[c, :n_exports] = 1
    
    return M

def generate_linear_nested_dense(n_countries=60, n_products=80):
    """
    Perfect linear nested structure, as dense as possible.
    Country c exports products 0...(c+extra) where extra makes it dense.
    """
    M = np.zeros((n_countries, n_products))
    
    for c in range(n_countries):
        # Linear scaling: country c exports first (c/60)*80 products
        n_exports = int((c+1) / n_countries * n_products)
        M[c, :n_exports] = 1
    
    return M

print("="*70)
print("TESTING DENSE NESTED STRUCTURES FOR λ₂^T ≈ 1")
print("="*70)

tests = [
    ("1. Sparse triangular (38% density)", generate_dense_nested_smooth(density=0.38)),
    ("2. Medium density nested (60%)", generate_dense_nested_smooth(density=0.6)),
    ("3. High density nested (80%)", generate_dense_nested_smooth(density=0.8)),
    ("4. Very dense nested (90%)", generate_very_dense_nested(density=0.9)),
    ("5. Linear perfect nested", generate_linear_nested_dense()),
]

for name, M in tests:
    eigvals, T = compute_eigenvalues(M)
    
    print(f"\n{name}")
    print("-"*70)
    print(f"  Density: {M.sum()/M.size:.2%}")
    print(f"  λ₁^T: {eigvals[0]:.6f}")
    print(f"  λ₂^T: {eigvals[1]:.6f}  ← Target: ≈1 for nested")
    print(f"  λ₃^T: {eigvals[2]:.6f}")
    print(f"  Eigengap: λ₂^T - λ₃^T = {eigvals[1] - eigvals[2]:.6f}")
    print(f"  Spectral gap: 1 - λ₂^T = {1 - eigvals[1]:.6f}")
    
    # Check country-country similarity structure
    # For nested, nearby countries should have high T[c,c'] values
    print(f"  Country similarity (diagonal band):")
    diag_similarity = np.mean([T[i, min(i+1, T.shape[0]-1)] for i in range(T.shape[0]-1)])
    distant_similarity = np.mean([T[i, min(i+20, T.shape[0]-1)] for i in range(T.shape[0]-20)])
    print(f"    Adjacent countries (avg T[c,c+1]): {diag_similarity:.4f}")
    print(f"    Distant countries (avg T[c,c+20]): {distant_similarity:.4f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nDid DENSE nested structures achieve λ₂^T ≈ 1?")
print("If NO: The paper's 'perfectly nested → λ₂^T ≈ 1' may be:")
print("  • A theoretical limit (infinite/continuous case)")
print("  • Referring to a different matrix (not our T)")
print("  • Or our generators still don't capture 'true' nestedness")
print("="*70)
