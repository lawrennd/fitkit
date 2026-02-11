#!/usr/bin/env python3
"""
Test perfect nested network to confirm theory.

Perfect nesting: If country c has capability θ_c and product p has complexity θ_p,
then M[c,p] = 1 iff θ_c ≥ θ_p

This creates a triangular matrix structure (after sorting).
Theory predicts: λ₂^T ≈ 1, λ₂^L ≈ 0
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

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

def generate_perfect_nested(n_countries=60, n_products=80, density_target=0.5):
    """Generate PERFECT nested network.
    
    Perfect nesting: M[c,p] = 1 iff capability[c] ≥ complexity[p]
    
    Args:
        density_target: Controls threshold to achieve target density
    """
    # Linearly spaced capabilities/complexities for perfect nesting
    capability = np.linspace(0, 1, n_countries)
    complexity = np.linspace(0, 1, n_products)
    
    # Create perfect triangular structure
    M = np.zeros((n_countries, n_products))
    for c in range(n_countries):
        for p in range(n_products):
            M[c, p] = 1 if capability[c] >= complexity[p] else 0
    
    return M

def generate_perfect_nested_dense(n_countries=60, n_products=80):
    """Generate PERFECT dense nested network.
    
    To match ~50% density, we create a triangular structure where:
    - Most capable countries export almost everything
    - Least capable export almost nothing
    - Creates smooth gradient
    """
    # For ~50% density with triangular structure, we need threshold ~0.5
    M = np.zeros((n_countries, n_products))
    
    for c in range(n_countries):
        # Country capability: 0 (least) to 1 (most)
        capability = c / (n_countries - 1)
        
        for p in range(n_products):
            # Product complexity: 0 (simplest) to 1 (most complex)
            complexity = p / (n_products - 1)
            
            # Perfect nesting: can produce if capability ≥ complexity
            if capability >= complexity:
                M[c, p] = 1
    
    return M

print("="*80)
print("TESTING PERFECT NESTED NETWORKS")
print("="*80)

# Test 1: Perfect nesting
print("\n1. PERFECT TRIANGULAR NESTING")
print("-"*80)
M = generate_perfect_nested_dense(n_countries=60, n_products=80)
eigvals_T, eigvals_L = compute_eigenvalues(M)

print(f"  Matrix size: {M.shape[0]} countries × {M.shape[1]} products")
print(f"  Density: {M.sum()/M.size:.2%}")
print(f"  Perfect structure: Every country c can produce product p iff c≥p (after sorting)")
print()
print(f"  λ₁^T (trivial): {eigvals_T[0]:.6f}")
print(f"  λ₂^T: {eigvals_T[1]:.6f}  ← Should be ≈1.0 for perfect nesting")
print(f"  λ₃^T: {eigvals_T[2]:.6f}")
print()
print(f"  λ₁^L (trivial): {eigvals_L[0]:.6f}")
print(f"  λ₂^L: {eigvals_L[1]:.6f}  ← Should be ≈0.0 for perfect nesting")
print(f"  λ₃^L: {eigvals_L[2]:.6f}")
print()
print(f"  Cheeger bound: Φ ≥ √(2λ₂^L) = {np.sqrt(2*eigvals_L[1]):.6f}")
print(f"  Gap: λ₃^L - λ₂^L = {eigvals_L[2] - eigvals_L[1]:.6f}")

# Test 2: Block diagonal (2 communities)
print("\n2. BLOCK DIAGONAL (2 COMMUNITIES)")
print("-"*80)
M_block = np.zeros((60, 80))
# Community 1: countries 0-29 export products 0-39
M_block[:30, :40] = 1
# Community 2: countries 30-59 export products 40-79
M_block[30:, 40:] = 1

eigvals_T_block, eigvals_L_block = compute_eigenvalues(M_block)
print(f"  Matrix size: {M_block.shape[0]} countries × {M_block.shape[1]} products")
print(f"  Density: {M_block.sum()/M_block.size:.2%}")
print(f"  Structure: 2 disjoint communities")
print()
print(f"  λ₁^T: {eigvals_T_block[0]:.6f}")
print(f"  λ₂^T: {eigvals_T_block[1]:.6f}  ← Should be <<1 for communities")
print(f"  λ₃^T: {eigvals_T_block[2]:.6f}")
print()
print(f"  λ₁^L: {eigvals_L_block[0]:.6f}")
print(f"  λ₂^L: {eigvals_L_block[1]:.6f}  ← Should be large for communities")
print(f"  λ₃^L: {eigvals_L_block[2]:.6f}")
print()
print(f"  Cheeger bound: Φ ≥ √(2λ₂^L) = {np.sqrt(2*eigvals_L_block[1]):.6f}")
print(f"  Gap: λ₃^L - λ₂^L = {eigvals_L_block[2] - eigvals_L_block[1]:.6f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nFrom Cheeger inequality and paper (line 448):")
print("  • Perfectly nested → λ₂^T ≈ 1, λ₂^L ≈ 0")
print("  • Communities → λ₂^T << 1, λ₂^L large")
print("\nThe principled diagnostic is:")
print("  • Compute λ₂^T from transition matrix T")
print("  • If λ₂^T > threshold (e.g., 0.8): 1D nested, skip community detection")
print("  • If λ₂^T < threshold: Multi-dimensional, run community detection")
print("="*80)
