#!/usr/bin/env python
"""
Quick test of the spectral-entropic comparison functionality.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from fitkit.algorithms.fitness import compute_fitness_complexity
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.stats import pearsonr

def compute_eci(M):
    """Compute ECI using second eigenvector of degree-normalized random walk."""
    M = M.astype(float)
    n_countries, n_products = M.shape
    
    k_c = M.sum(axis=1) + 1e-10
    k_p = M.sum(axis=0) + 1e-10
    
    D_c_inv = sparse.diags(1.0 / k_c)
    D_p_inv = sparse.diags(1.0 / k_p)
    M_sparse = sparse.csr_matrix(M)
    
    T_countries = D_c_inv @ M_sparse @ D_p_inv @ M_sparse.T
    
    n_eigs = min(3, n_countries - 1)
    if n_countries > 2:
        eigenvalues, eigenvectors = eigs(T_countries, k=n_eigs, which='LM')
        idx = np.argsort(np.real(eigenvalues))[::-1]
        eigenvectors = np.real(eigenvectors[:, idx])
        eci = eigenvectors[:, 1]
        eci = (eci - eci.mean()) / (eci.std() + 1e-10)
    else:
        eci = np.zeros(n_countries)
    
    return eci

def generate_nested_network(n_countries=30, n_products=40, seed=42):
    """Generate a nested network for testing."""
    np.random.seed(seed)
    capability = np.sort(np.random.uniform(0, 1, n_countries))
    complexity = np.sort(np.random.uniform(0, 1, n_products))
    
    M = np.zeros((n_countries, n_products))
    for c in range(n_countries):
        for p in range(n_products):
            if capability[c] >= complexity[p]:
                if np.random.random() > 0.1:
                    M[c, p] = 1
            elif np.random.random() < 0.05:
                M[c, p] = 1
    
    return M

def test_basic_functionality():
    """Test that the basic functionality works."""
    print("Generating test network...")
    M = generate_nested_network()
    print(f"Network shape: {M.shape}")
    print(f"Network density: {M.sum() / M.size:.2%}")
    
    print("\nComputing spectral measures (ECI)...")
    eci = compute_eci(M)
    print(f"ECI computed: {len(eci)} values")
    print(f"ECI range: [{eci.min():.3f}, {eci.max():.3f}]")
    
    print("\nComputing entropic measures (Fitness)...")
    fitness, complexity = compute_fitness_complexity(M)
    print(f"Fitness computed: {len(fitness)} values")
    print(f"Fitness range: [{fitness.min():.3f}, {fitness.max():.3f}]")
    
    # Standardize fitness
    fitness_std = (fitness - fitness.mean()) / (fitness.std() + 1e-10)
    
    print("\nComputing correlation...")
    r, p = pearsonr(eci, fitness_std)
    print(f"Pearson correlation (ECI vs Fitness): r = {r:.3f}, p = {p:.3e}")
    
    if r > 0.7:
        print("\n✓ SUCCESS: High correlation indicates nested hierarchy")
        print("  The spectral and entropic measures agree, as expected for a nested network.")
    else:
        print("\n? Moderate correlation")
        print(f"  Correlation = {r:.3f}")
    
    return True

if __name__ == '__main__':
    try:
        test_basic_functionality()
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
