#!/usr/bin/env python3
"""
Principled Cheeger-based diagnostic for nested vs community structure.

From economic-fitness.tex (lines 405-407, 453):
- L = I - T where T is the transition matrix
- λ_L = 1 - λ_T with same eigenvectors
- Cheeger: λ₂^L ≥ Φ²/2 where Φ is conductance
- Perfectly nested → λ₂^T ≈ 1 (equivalently λ₂^L ≈ 0)
- Communities → λ₂^T << 1 (equivalently λ₂^L large)
"""
import numpy as np
from scipy import sparse, linalg

def build_transition_matrix(M):
    """Build T = D_c^{-1} M D_p^{-1} M^T."""
    if sparse.issparse(M):
        M = M.toarray()
    
    D_c = M.sum(axis=1) + 1e-10
    D_p = M.sum(axis=0) + 1e-10
    
    M_weighted = M / D_p[np.newaxis, :]
    MM_T = M @ M_weighted.T
    T = MM_T / D_c[:, np.newaxis]
    
    return T

def compute_eigenvalues_proper(M):
    """Compute eigenvalues of T using scipy.linalg.eig for non-symmetric matrix."""
    T = build_transition_matrix(M)
    
    # T is generally not symmetric, so use eig (not eigh/eigsh)
    # However, T should have real eigenvalues (it's a transition-like matrix)
    eigvals_T, _ = linalg.eig(T)
    
    # Take real part (imaginary should be ~0) and sort descending
    eigvals_T = np.real(eigvals_T)
    eigvals_T = np.sort(eigvals_T)[::-1]
    
    # Compute Laplacian eigenvalues: λ_L = 1 - λ_T
    eigvals_L = 1.0 - eigvals_T
    
    return eigvals_T, eigvals_L, T

def generate_perfect_nested(n_countries=60, n_products=80):
    """Perfect triangular nesting: M[c,p] = 1 iff c ≥ p."""
    M = np.zeros((n_countries, n_products))
    for c in range(n_countries):
        for p in range(n_products):
            if c >= p:
                M[c, p] = 1
    return M

def generate_block_diagonal(n_countries=60, n_products=80):
    """2 disjoint communities."""
    M = np.zeros((n_countries, n_products))
    M[:30, :40] = 1
    M[30:, 40:] = 1
    return M

def cheeger_diagnostic(M, name="Network"):
    """Apply Cheeger-based diagnostic."""
    eigvals_T, eigvals_L, T = compute_eigenvalues_proper(M)
    
    print(f"\n{name}")
    print("-" * 70)
    print(f"  Density: {M.sum()/M.size:.2%}")
    print(f"  Matrix eigenvalue properties:")
    print(f"    T is symmetric: {np.allclose(T, T.T)}")
    print(f"    Max imaginary part: {np.max(np.abs(np.imag(linalg.eig(T)[0]))):.2e}")
    print()
    print(f"  Transition matrix eigenvalues (T):")
    print(f"    λ₁^T (trivial): {eigvals_T[0]:.6f}  (should be ≈1)")
    print(f"    λ₂^T: {eigvals_T[1]:.6f}  ← KEY: ≈1 for nested, <<1 for communities")
    print(f"    λ₃^T: {eigvals_T[2]:.6f}")
    print()
    print(f"  Laplacian eigenvalues (L = I - T):")
    print(f"    λ₁^L (trivial): {eigvals_L[0]:.6f}  (should be ≈0)")
    print(f"    λ₂^L: {eigvals_L[1]:.6f}  ← KEY: ≈0 for nested, large for communities")
    print(f"    λ₃^L: {eigvals_L[2]:.6f}")
    print()
    print(f"  Cheeger bound: Φ ≥ √(2λ₂^L) = {np.sqrt(2 * max(0, eigvals_L[1])):.6f}")
    print(f"  Spectral gap: λ₃^L - λ₂^L = {eigvals_L[2] - eigvals_L[1]:.6f}")
    print()
    
    # Diagnostic
    if eigvals_T[1] > 0.8:
        print(f"  DIAGNOSIS: λ₂^T = {eigvals_T[1]:.3f} > 0.8")
        print(f"            → 1D nested hierarchy (high conductance, no bottlenecks)")
        print(f"            → Community detection NOT appropriate")
    else:
        print(f"  DIAGNOSIS: λ₂^T = {eigvals_T[1]:.3f} < 0.8")
        print(f"            → Multi-dimensional structure (low conductance, bottlenecks)")
        print(f"            → Community detection appropriate")
    
    return eigvals_T[1], eigvals_L[1]

print("=" * 70)
print("CHEEGER-BASED DIAGNOSTIC FOR NETWORK STRUCTURE")
print("=" * 70)
print("\nFrom economic-fitness.tex:")
print("  • Perfectly nested → λ₂^T ≈ 1 (equivalently λ₂^L ≈ 0)")
print("  • Communities → λ₂^T << 1 (equivalently λ₂^L large)")
print("  • Cheeger inequality: λ₂^L ≥ Φ²/2 (Φ = conductance)")

# Test 1: Perfect nesting
M_nested = generate_perfect_nested(60, 80)
lambda2_T_nested, lambda2_L_nested = cheeger_diagnostic(M_nested, "1. PERFECT TRIANGULAR NESTING")

# Test 2: Block diagonal (communities)
M_block = generate_block_diagonal(60, 80)
lambda2_T_block, lambda2_L_block = cheeger_diagnostic(M_block, "2. BLOCK DIAGONAL (2 COMMUNITIES)")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nPerfectly nested:  λ₂^T = {lambda2_T_nested:.4f}, λ₂^L = {lambda2_L_nested:.4f}")
print(f"Communities:       λ₂^T = {lambda2_T_block:.4f}, λ₂^L = {lambda2_L_block:.4f}")
print("\nThe principled diagnostic:")
print("  if λ₂^T > 0.8:  # High persistence, no bottleneck")
print("      → 1D nested, skip community detection")
print("  else:           # Low persistence, bottlenecks present")
print("      → Multi-dimensional, run community detection")
print("=" * 70)
