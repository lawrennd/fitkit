#!/usr/bin/env python3
"""
Principled Cheeger-based diagnostic using TWO criteria.

From economic-fitness.tex lines 446 and 453, there are TWO gaps:

1. Eigengap: λ₂^T - λ₃^T (dimensionality)
   - Large gap → 1D dominant mode (nested)
   - Small gap → Multiple significant modes (communities)

2. Spectral gap: 1 - λ₂^T = λ₂^L (conductance via Cheeger)
   - Small (λ₂^T ≈ 1, λ₂^L ≈ 0) → High persistence, slow mixing along 1D gradient
   - Large (λ₂^T << 1, λ₂^L large) → Fast local mixing, bottlenecks between communities

For nested networks:
  • λ₂^T ≈ 1 (persistent mode)
  • λ₂^T >> λ₃^T (1D dominant)
  • The "bottleneck" is the capability gradient itself (1D structure)

For community networks:
  • λ₂^T, λ₃^T, ... all < 1 (multiple modes)
  • λ₂^T ≈ λ₃^T (multi-dimensional)
  • Bottlenecks BETWEEN communities (disconnected structure)
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

def compute_principled_diagnostic(M):
    """
    Compute principled Cheeger diagnostic using TWO criteria.
    
    Returns:
        dict with eigenvalues, gaps, and diagnostic
    """
    T = build_transition_matrix(M)
    eigvals_T, _ = linalg.eig(T)
    eigvals_T = np.real(eigvals_T)
    eigvals_T = np.sort(eigvals_T)[::-1]
    
    # Key eigenvalues
    lambda1_T = eigvals_T[0]  # Should be ≈1 (trivial)
    lambda2_T = eigvals_T[1]  # Second largest (persistence)
    lambda3_T = eigvals_T[2]  # Third largest
    
    # Laplacian eigenvalues
    lambda2_L = 1.0 - lambda2_T
    
    # Two gaps
    eigengap = lambda2_T - lambda3_T  # Dimensionality gap
    spectral_gap = 1.0 - lambda2_T    # Cheeger gap (= λ₂^L)
    
    # Diagnostic thresholds (from paper + empirics)
    PERSISTENCE_THRESHOLD = 0.8  # λ₂^T > 0.8 → persistent mode
    EIGENGAP_THRESHOLD = 0.3     # λ₂^T - λ₃^T > 0.3 → 1D dominant
    
    # Principled diagnostic
    is_persistent = lambda2_T > PERSISTENCE_THRESHOLD
    is_1d = eigengap > EIGENGAP_THRESHOLD
    
    if is_persistent and is_1d:
        diagnosis = "1D nested hierarchy"
        recommendation = "Community detection NOT appropriate (single capability gradient)"
    elif not is_persistent or not is_1d:
        diagnosis = "Multi-dimensional structure"
        recommendation = "Community detection appropriate (multiple modes/communities)"
    else:
        diagnosis = "Ambiguous"
        recommendation = "Edge case - manual inspection recommended"
    
    return {
        'lambda1_T': lambda1_T,
        'lambda2_T': lambda2_T,
        'lambda3_T': lambda3_T,
        'lambda2_L': lambda2_L,
        'eigengap': eigengap,
        'spectral_gap': spectral_gap,
        'is_persistent': is_persistent,
        'is_1d': is_1d,
        'diagnosis': diagnosis,
        'recommendation': recommendation
    }

def generate_nested_bipartite(n_countries=60, n_products=80, noise=0.01):
    """
    Generate nested bipartite network.
    True nesting: M[c,p] = 1 iff capability[c] >= complexity[p]
    """
    np.random.seed(42)
    capability = np.linspace(0, 1, n_countries)
    complexity = np.linspace(0, 1, n_products)
    
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

def generate_community_bipartite(n_countries=60, n_products=80, n_communities=3):
    """Generate bipartite with clear community structure."""
    np.random.seed(42)
    M = np.zeros((n_countries, n_products))
    
    countries_per_comm = n_countries // n_communities
    products_per_comm = n_products // n_communities
    
    for comm in range(n_communities):
        c_start = comm * countries_per_comm
        c_end = (comm + 1) * countries_per_comm if comm < n_communities - 1 else n_countries
        
        p_start = comm * products_per_comm
        p_end = (comm + 1) * products_per_comm if comm < n_communities - 1 else n_products
        
        # Dense within community
        M[c_start:c_end, p_start:p_end] = 1
        
        # Sparse between communities (10% inter-community links)
        if comm < n_communities - 1:
            n_inter = int(0.1 * (c_end - c_start) * (p_end - p_start))
            for _ in range(n_inter):
                c = np.random.randint(c_start, c_end)
                p = np.random.randint(p_end, n_products)
                M[c, p] = 1
    
    return M

def print_diagnostic_report(name, M):
    """Print comprehensive diagnostic report."""
    result = compute_principled_diagnostic(M)
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Density: {M.sum()/M.size:.2%}")
    print()
    print("Eigenvalues (T):")
    print(f"  λ₁^T: {result['lambda1_T']:.6f}  (trivial, should be ≈1)")
    print(f"  λ₂^T: {result['lambda2_T']:.6f}  (persistence)")
    print(f"  λ₃^T: {result['lambda3_T']:.6f}")
    print()
    print("Laplacian:")
    print(f"  λ₂^L: {result['lambda2_L']:.6f}  (Cheeger conductance bound)")
    print()
    print("Two Gaps:")
    print(f"  Eigengap (λ₂^T - λ₃^T): {result['eigengap']:.6f}  {'✓ Large (1D)' if result['is_1d'] else '✗ Small (multi-D)'}")
    print(f"  Spectral gap (1 - λ₂^T): {result['spectral_gap']:.6f}  {'✓ Small (persistent)' if result['is_persistent'] else '✗ Large (transient)'}")
    print()
    print(f"DIAGNOSIS: {result['diagnosis']}")
    print(f"  → {result['recommendation']}")
    print()
    print("Interpretation:")
    if result['is_persistent'] and result['is_1d']:
        print("  • Single persistent mode (λ₂^T ≈ 1) dominates")
        print("  • Higher modes negligible (large eigengap)")
        print("  • Slow mixing along 1D capability gradient (nested hierarchy)")
        print("  • The 'bottleneck' IS the gradient itself, not separate communities")
    else:
        print("  • Multiple significant modes (λ₂^T, λ₃^T both < 1)")
        print("  • Multi-dimensional structure (small eigengap)")
        print("  • Bottlenecks BETWEEN distinct communities")
        print("  • Fast local mixing within, slow mixing between")

print("="*70)
print("PRINCIPLED CHEEGER DIAGNOSTIC: TWO-CRITERION APPROACH")
print("="*70)
print()
print("Criteria:")
print("  1. Persistence: λ₂^T > 0.8")
print("  2. Dimensionality: λ₂^T - λ₃^T > 0.3")
print()
print("Nested → BOTH criteria met")
print("Communities → At least one criterion fails")

# Test 1: Nested network (1% noise)
M_nested = generate_nested_bipartite(noise=0.01)
print_diagnostic_report("1. NESTED NETWORK (1% noise)", M_nested)

# Test 2: Nested network (5% noise)
M_nested_noisy = generate_nested_bipartite(noise=0.05)
print_diagnostic_report("2. NESTED NETWORK (5% noise)", M_nested_noisy)

# Test 3: Community structure (3 communities)
M_communities = generate_community_bipartite(n_communities=3)
print_diagnostic_report("3. COMMUNITY STRUCTURE (3 communities)", M_communities)

# Test 4: Block diagonal (extreme communities)
M_block = np.zeros((60, 80))
M_block[:30, :40] = 1
M_block[30:, 40:] = 1
print_diagnostic_report("4. BLOCK DIAGONAL (2 disjoint communities)", M_block)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print()
print("The principled diagnostic uses BOTH:")
print("  1. λ₂^T magnitude (persistence via Cheeger)")
print("  2. λ₂^T - λ₃^T gap (dimensionality)")
print()
print("This resolves the apparent contradiction:")
print("  • Nested: λ₂^T ≈ 1 AND λ₂^T >> λ₃^T")
print("  • Communities: λ₂^T < 1 OR λ₂^T ≈ λ₃^T")
print()
print("Cheeger's inequality λ₂^L ≥ Φ²/2 applies to BOTH cases,")
print("but the INTERPRETATION differs:")
print("  • Nested: Small λ₂^L → high conductance → slow mixing along 1D gradient")
print("  • Communities: Large λ₂^L → low conductance → bottlenecks BETWEEN groups")
print("="*70)
