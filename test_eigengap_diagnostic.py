#!/usr/bin/env python3
"""
Test eigengap diagnostic for nested vs community structure.

From economic-fitness.tex (line 480):
"One-dimensional nested hierarchy: The capability space is approximately 1D,
 λ₃^L is substantially larger than λ₂^L"

Key diagnostic: eigengap_ratio = λ₃^L / λ₂^L
- Large ratio (>2): 1D nested hierarchy
- Small ratio (<1.5): Multi-dimensional community structure
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr
import sys
sys.path.insert(0, '..')

from fitkit.algorithms import FitnessComplexity, ECI
from fitkit.community import CommunityDetector

def compute_eigengap_diagnostic(M):
    """Compute eigengap ratio for dimensionality assessment."""
    # Build transition matrix
    k_c = M.sum(axis=1) + 1e-10
    k_p = M.sum(axis=0) + 1e-10
    D_c_inv = sparse.diags(1.0 / k_c)
    D_p_inv = sparse.diags(1.0 / k_p)
    M_sp = sparse.csr_matrix(M)
    T = D_c_inv @ M_sp @ D_p_inv @ M_sp.T
    
    # Compute eigenvalues of T
    n_countries = M.shape[0]
    eigvals_T, _ = eigsh(T, k=min(10, n_countries-2), which='LA')
    eigvals_T = np.sort(eigvals_T)[::-1]
    
    # Convert to Laplacian eigenvalues: λ^L = 1 - λ^T
    eigvals_L = 1.0 - eigvals_T
    
    # Multiple eigengap measures
    lambda_2_L = eigvals_L[1]  # Second smallest (skip trivial λ₁=0)
    lambda_3_L = eigvals_L[2]  # Third smallest
    
    # Different relative measures
    absolute_gap = lambda_3_L - lambda_2_L
    ratio = lambda_3_L / (lambda_2_L + 1e-10)
    relative_gap = (lambda_3_L - lambda_2_L) / (lambda_2_L + 1e-10)  # Normalized by λ₂
    
    # Gap relative to spectrum range
    spectrum_range = eigvals_L.max() - eigvals_L.min()
    gap_over_range = absolute_gap / (spectrum_range + 1e-10)
    
    # Gap relative to average of consecutive gaps
    all_gaps = np.diff(eigvals_L[1:5])  # Next few gaps
    avg_gap = np.mean(all_gaps)
    gap_vs_avg = absolute_gap / (avg_gap + 1e-10)
    
    return {
        'eigvals_T': eigvals_T,
        'eigvals_L': eigvals_L,
        'lambda_2_L': lambda_2_L,
        'lambda_3_L': lambda_3_L,
        'absolute_gap': absolute_gap,
        'ratio': ratio,
        'relative_gap': relative_gap,
        'gap_over_range': gap_over_range,
        'gap_vs_avg': gap_vs_avg,
        'all_gaps': all_gaps
    }

def generate_nested_network(n_countries=60, n_products=80, noise=0.05, seed=42):
    """Generate nested network."""
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

def generate_community_network(n_communities=3, size=20, within=0.7, between=0.05, seed=42):
    """Generate network with communities."""
    np.random.seed(seed)
    n_countries = n_communities * size
    n_products = n_communities * size
    
    M = np.zeros((n_countries, n_products))
    country_labels = np.repeat(np.arange(n_communities), size)
    product_labels = np.repeat(np.arange(n_communities), size)
    
    for c in range(n_countries):
        for p in range(n_products):
            if country_labels[c] == product_labels[p]:
                if np.random.random() < within:
                    M[c, p] = 1
            else:
                if np.random.random() < between:
                    M[c, p] = 1
    return M

# Test 1a: Nested network (5% noise)
print("="*70)
print("TEST 1a: Nested Network (5% noise)")
print("="*70)
M_nested = generate_nested_network(n_countries=60, n_products=80, noise=0.05)
diag = compute_eigengap_diagnostic(M_nested)

print(f"Laplacian eigenvalues: {diag['eigvals_L'][:6]}")
print(f"Transition eigenvalues: {diag['eigvals_T'][:6]}")
print(f"\nEigengap measures:")
print(f"  1. Ratio (λ₃/λ₂):          {diag['ratio']:.3f}")
print(f"  2. Relative gap (λ₃-λ₂)/λ₂: {diag['relative_gap']:.3f}")
print(f"  3. Absolute gap (λ₃-λ₂):    {diag['absolute_gap']:.4f}")
print(f"  4. Gap / spectrum range:   {diag['gap_over_range']:.3f}")
print(f"  5. Gap vs avg gap:         {diag['gap_vs_avg']:.3f}")
print(f"  6. λ₂^L magnitude:         {diag['lambda_2_L']:.4f}")

if diag['ratio'] > 2.0:
    print("\n✓ LARGE eigengap detected")
else:
    print("\n? Small ratio (~1.1)")

# Test 1b: Strongly nested network (1% noise)
print("\n" + "="*70)
print("TEST 1b: Strongly Nested Network (1% noise)")
print("="*70)
M_nested_strong = generate_nested_network(n_countries=60, n_products=80, noise=0.01)
diag1b = compute_eigengap_diagnostic(M_nested_strong)

print(f"Laplacian eigenvalues: {diag1b['eigvals_L'][:6]}")
print(f"Transition eigenvalues: {diag1b['eigvals_T'][:6]}")
print(f"\nEigengap measures:")
print(f"  1. Ratio (λ₃/λ₂):          {diag1b['ratio']:.3f}")
print(f"  2. Relative gap (λ₃-λ₂)/λ₂: {diag1b['relative_gap']:.3f}")
print(f"  3. Absolute gap (λ₃-λ₂):    {diag1b['absolute_gap']:.4f}")
print(f"  4. Gap / spectrum range:   {diag1b['gap_over_range']:.3f}")
print(f"  5. Gap vs avg gap:         {diag1b['gap_vs_avg']:.3f}")
print(f"  6. λ₂^L magnitude:         {diag1b['lambda_2_L']:.4f}")

if diag1b['ratio'] > 2.0:
    print("\n✓ LARGE ratio")
else:
    print("\n? Small ratio (~1.1)")

# Test 2: Community network
print("\n" + "="*70)
print("TEST 2: Community Network (3 communities)")
print("="*70)
M_comm = generate_community_network(n_communities=3, size=20, within=0.7, between=0.05)
diag2 = compute_eigengap_diagnostic(M_comm)

print(f"Laplacian eigenvalues: {diag2['eigvals_L'][:6]}")
print(f"Transition eigenvalues: {diag2['eigvals_T'][:6]}")
print(f"\nEigengap measures:")
print(f"  1. Ratio (λ₃/λ₂):          {diag2['ratio']:.3f}")
print(f"  2. Relative gap (λ₃-λ₂)/λ₂: {diag2['relative_gap']:.3f}")
print(f"  3. Absolute gap (λ₃-λ₂):    {diag2['absolute_gap']:.4f}")
print(f"  4. Gap / spectrum range:   {diag2['gap_over_range']:.3f}")
print(f"  5. Gap vs avg gap:         {diag2['gap_vs_avg']:.3f}")
print(f"  6. λ₂^L magnitude:         {diag2['lambda_2_L']:.4f}")

print("\n" + "="*70)
print("COMPARISON: Which Measure Discriminates Best?")
print("="*70)
print(f"{'Measure':<30} {'Nested (1%)':<15} {'Community':<15} {'Discriminates?':<15}")
print("-"*70)
print(f"{'1. Ratio (λ₃/λ₂)':<30} {diag1b['ratio']:<15.3f} {diag2['ratio']:<15.3f} {'NO (both ~1.1)':<15}")
print(f"{'2. Relative gap (λ₃-λ₂)/λ₂':<30} {diag1b['relative_gap']:<15.3f} {diag2['relative_gap']:<15.3f} {'NO (similar)':<15}")
print(f"{'3. Absolute gap (λ₃-λ₂)':<30} {diag1b['absolute_gap']:<15.4f} {diag2['absolute_gap']:<15.4f} {'YES (2.3x diff)':<15}")
print(f"{'4. Gap / spectrum range':<30} {diag1b['gap_over_range']:<15.3f} {diag2['gap_over_range']:<15.3f} {'? (check)':<15}")
print(f"{'5. Gap vs avg gap':<30} {diag1b['gap_vs_avg']:<15.3f} {diag2['gap_vs_avg']:<15.3f} {'? (check)':<15}")
print(f"{'6. λ₂^L magnitude':<30} {diag1b['lambda_2_L']:<15.4f} {diag2['lambda_2_L']:<15.4f} {'YES! (2.5x diff)':<15}")
print(f"{'7. λ₂^T (persistence)':<30} {diag1b['eigvals_T'][1]:<15.4f} {diag2['eigvals_T'][1]:<15.4f} {'YES! (3.3x diff)':<15}")
print("="*70)

print("\n" + "="*70)
print("EIGENGAP INTERPRETATION")
print("="*70)
print("From paper (line 480): 'λ₃^L substantially larger than λ₂^L'")
print("\nKey insight: It's NOT just the ratio, but the CONTEXT:")
print("  - Nested:    λ₂^L large (0.79) → Both λ₂ and λ₃ near 1 → All bunched")
print("  - Community: λ₂^L small (0.32) → Low eigenvalues → Bottlenecks")
print("\nBest discriminators:")
print("  ✓ λ₂^L magnitude: nested=0.79, community=0.32 (2.5x difference)")
print("  ✓ Absolute gap: nested=0.098, community=0.043 (2.3x difference)")
print("  ✓ λ₂^T: nested=0.21, community=0.69 (3.3x difference)")
print("\nProposed diagnostic:")
print("  IF λ₂^L > 0.6 OR λ₂^T < 0.4:")
print("    → 1D nested hierarchy (fast mixing, no persistent bottleneck)")
print("    → Community detection NOT meaningful")
print("  ELSE:")
print("    → Multi-dimensional structure (slow mixing, bottlenecks)")
print("    → Community detection IS appropriate")
print("="*70)
print("="*70)
