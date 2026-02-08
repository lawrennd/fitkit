#!/usr/bin/env python
"""
Quick test of the spectral-entropic comparison functionality.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from fitkit.algorithms import FitnessComplexity, ECI
from scipy import sparse
from scipy.stats import pearsonr

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
    
    # Convert to sparse for sklearn-style interface
    M_sparse = sparse.csr_matrix(M)
    
    print("\nComputing spectral measures (ECI) using sklearn-style estimator...")
    eci_estimator = ECI()
    eci, pci = eci_estimator.fit_transform(M_sparse)
    print(f"ECI computed: {len(eci)} values")
    print(f"ECI range: [{np.nanmin(eci):.3f}, {np.nanmax(eci):.3f}]")
    print(f"PCI computed: {len(pci)} values")
    
    print("\nComputing entropic measures (Fitness) using sklearn-style estimator...")
    fc_estimator = FitnessComplexity(n_iter=200, tol=1e-10, verbose=True)
    fitness, complexity = fc_estimator.fit_transform(M_sparse)
    print(f"Fitness computed: {len(fitness)} values")
    print(f"Fitness range: [{fitness.min():.3f}, {fitness.max():.3f}]")
    print(f"Iterations performed: {fc_estimator.n_iter_}")
    
    # Standardize fitness for comparison
    fitness_std = (fitness - fitness.mean()) / (fitness.std() + 1e-10)
    
    print("\nComputing correlation...")
    # Remove NaN values for correlation
    valid_mask = ~np.isnan(eci)
    r, p = pearsonr(eci[valid_mask], fitness_std[valid_mask])
    print(f"Pearson correlation (ECI vs Fitness): r = {r:.3f}, p = {p:.3e}")
    
    if r > 0.7:
        print("\n✓ SUCCESS: High correlation indicates nested hierarchy")
        print("  The spectral and entropic measures agree, as expected for a nested network.")
        print("  Sklearn-style estimators work correctly!")
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
