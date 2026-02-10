#!/usr/bin/env python3
"""
Test with REALISTIC nested structure.

The issue: My "perfect triangular" matrix (c ≥ p) gives λ₂^T = 0.25, not ≈1.
This suggests it's not actually a "well-nested" structure in the paper's sense.

Real nested networks have a SMOOTH capability gradient, not a sharp cutoff.
Let me try probabilistic nesting with smooth transitions.
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

def generate_smooth_nested(n_countries=60, n_products=80, sharpness=10.0, noise=0.05):
    """
    Generate smooth nested structure using sigmoid.
    
    For each (country, product) pair:
    - country c has capability θ_c ∈ [0, 1]
    - product p has complexity κ_p ∈ [0, 1]
    - Probability M[c,p] = 1 is sigmoid(sharpness * (θ_c - κ_p))
    
    Args:
        sharpness: How sharp the transition is (high = more like perfect triangular)
        noise: Base noise level
    """
    np.random.seed(42)
    
    # Smooth capability/complexity gradients
    capability = np.linspace(0, 1, n_countries)
    complexity = np.linspace(0, 1, n_products)
    
    M = np.zeros((n_countries, n_products))
    for c in range(n_countries):
        for p in range(n_products):
            # Sigmoid transition: high prob if capability >> complexity
            gap = capability[c] - complexity[p]
            prob = 1.0 / (1.0 + np.exp(-sharpness * gap))
            prob = (1 - noise) * prob + noise * 0.5  # Add noise
            M[c, p] = 1 if np.random.random() < prob else 0
    
    return M

def generate_dense_smooth_nested(n_countries=60, n_products=80, sharpness=5.0):
    """
    Generate dense smooth nested without noise.
    Higher density ~ more connections ~ better diffusion.
    """
    capability = np.linspace(0, 1, n_countries)
    complexity = np.linspace(0, 1, n_products)
    
    M = np.zeros((n_countries, n_products))
    for c in range(n_countries):
        for p in range(n_products):
            gap = capability[c] - complexity[p]
            prob = 1.0 / (1.0 + np.exp(-sharpness * gap))
            M[c, p] = 1 if prob > 0.5 else 0
    
    return M

def generate_full_random(n_countries=60, n_products=80, density=0.5):
    """Completely random (no structure)."""
    np.random.seed(42)
    M = (np.random.random((n_countries, n_products)) < density).astype(float)
    return M

print("=" * 70)
print("TESTING DIFFERENT NESTED STRUCTURES")
print("=" * 70)

# Test 1: Sharp threshold (my original "perfect" nested)
print("\n1. SHARP THRESHOLD (c ≥ p)")
print("-" * 70)
M_sharp = np.zeros((60, 80))
for c in range(60):
    for p in range(80):
        M_sharp[c, p] = 1 if c >= p else 0
eigvals_sharp = compute_eigenvalues(M_sharp)
print(f"  Density: {M_sharp.sum()/M_sharp.size:.2%}")
print(f"  λ₁^T: {eigvals_sharp[0]:.4f}")
print(f"  λ₂^T: {eigvals_sharp[1]:.4f}  ← Target: ≈1 for nested")
print(f"  λ₃^T: {eigvals_sharp[2]:.4f}")

# Test 2: Smooth sigmoid (sharpness=10)
print("\n2. SMOOTH SIGMOID (sharpness=10, noise=5%)")
print("-" * 70)
M_smooth10 = generate_smooth_nested(sharpness=10.0, noise=0.05)
eigvals_smooth10 = compute_eigenvalues(M_smooth10)
print(f"  Density: {M_smooth10.sum()/M_smooth10.size:.2%}")
print(f"  λ₁^T: {eigvals_smooth10[0]:.4f}")
print(f"  λ₂^T: {eigvals_smooth10[1]:.4f}  ← Target: ≈1 for nested")
print(f"  λ₃^T: {eigvals_smooth10[2]:.4f}")

# Test 3: Smoother sigmoid (sharpness=5)
print("\n3. SMOOTHER SIGMOID (sharpness=5, noise=5%)")
print("-" * 70)
M_smooth5 = generate_smooth_nested(sharpness=5.0, noise=0.05)
eigvals_smooth5 = compute_eigenvalues(M_smooth5)
print(f"  Density: {M_smooth5.sum()/M_smooth5.size:.2%}")
print(f"  λ₁^T: {eigvals_smooth5[0]:.4f}")
print(f"  λ₂^T: {eigvals_smooth5[1]:.4f}  ← Target: ≈1 for nested")
print(f"  λ₃^T: {eigvals_smooth5[2]:.4f}")

# Test 4: Dense smooth (no noise, deterministic)
print("\n4. DENSE SMOOTH SIGMOID (sharpness=5, no noise)")
print("-" * 70)
M_dense = generate_dense_smooth_nested(sharpness=5.0)
eigvals_dense = compute_eigenvalues(M_dense)
print(f"  Density: {M_dense.sum()/M_dense.size:.2%}")
print(f"  λ₁^T: {eigvals_dense[0]:.4f}")
print(f"  λ₂^T: {eigvals_dense[1]:.4f}  ← Target: ≈1 for nested")
print(f"  λ₃^T: {eigvals_dense[2]:.4f}")

# Test 5: Very dense, very smooth
print("\n5. VERY DENSE & SMOOTH (sharpness=2, no noise)")
print("-" * 70)
M_verysmooth = generate_dense_smooth_nested(sharpness=2.0)
eigvals_verysmooth = compute_eigenvalues(M_verysmooth)
print(f"  Density: {M_verysmooth.sum()/M_verysmooth.size:.2%}")
print(f"  λ₁^T: {eigvals_verysmooth[0]:.4f}")
print(f"  λ₂^T: {eigvals_verysmooth[1]:.4f}  ← Target: ≈1 for nested")
print(f"  λ₃^T: {eigvals_verysmooth[2]:.4f}")

# Test 6: Random (no structure)
print("\n6. COMPLETELY RANDOM (no structure)")
print("-" * 70)
M_random = generate_full_random(density=0.5)
eigvals_random = compute_eigenvalues(M_random)
print(f"  Density: {M_random.sum()/M_random.size:.2%}")
print(f"  λ₁^T: {eigvals_random[0]:.4f}")
print(f"  λ₂^T: {eigvals_random[1]:.4f}")
print(f"  λ₃^T: {eigvals_random[2]:.4f}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
print("\nNone of the 'nested' structures give λ₂^T ≈ 1!")
print("This suggests:")
print("  1. The triangular structure itself creates bottlenecks")
print("  2. Real 'nested' networks in the paper might be much denser")
print("  3. Or the paper's 'perfectly nested' is a theoretical limit,")
print("     not achievable with finite sparse matrices")
print("\nNext: Check if DENSE (almost complete) bipartite gives λ₂^T ≈ 1")
print("=" * 70)
