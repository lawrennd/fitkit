"""Tests for Fitness-Complexity filtering functionality."""

import numpy as np
import scipy.sparse as sp
import pytest
from fitkit.algorithms import FitnessComplexity


def test_fitness_filtering_unique_product():
    """Test that low-ubiquity products are filtered out."""
    # Create matrix where country 0 has products with low ubiquity
    M = sp.lil_matrix((10, 10), dtype=float)
    M[0, 0] = 1  # Product 0: ubiquity=1
    M[0, 1] = 1  # Product 1: ubiquity will be 2
    M[1, 1] = 1
    
    # Fill rest with random structure
    np.random.seed(42)
    for i in range(10):
        for j in range(2, 10):
            if np.random.rand() > 0.7:
                M[i, j] = 1
    
    M = M.tocsr()
    
    # Run with default filtering (min_ubiquity=3)
    fc = FitnessComplexity(min_ubiquity=3, min_diversification=2, verbose=False)
    F, Q = fc.fit_transform(M)
    
    # Check that products 0 and 1 were filtered (ubiquity < 3)
    assert fc.product_mask_[0] == False, "Product 0 should be filtered (ubiquity=1)"
    assert fc.product_mask_[1] == False, "Product 1 should be filtered (ubiquity=2)"
    assert fc.n_products_filtered_ >= 2, "At least two products should be filtered"
    
    # Check that fitness is not dominated by country 0
    F_valid = F[~np.isnan(F)]
    if len(F_valid) > 1:
        max_fitness = F_valid.max()
        median_fitness = np.median(F_valid)
        assert max_fitness / median_fitness < 50, "Fitness should not be extremely concentrated"


def test_fitness_filtering_low_diversification():
    """Test that countries with low diversification are filtered out."""
    # Create matrix where country 0 exports only 1 product
    M = sp.lil_matrix((10, 10), dtype=float)
    M[0, 0] = 1  # Country 0: only 1 product
    
    # Other countries export many products
    for i in range(1, 10):
        for j in range(10):
            if np.random.rand() > 0.3:
                M[i, j] = 1
    
    M = M.tocsr()
    
    # Run with default filtering (min_diversification=5)
    fc = FitnessComplexity(min_ubiquity=1, min_diversification=5, verbose=False)
    F, Q = fc.fit_transform(M)
    
    # Check that country 0 was filtered (diversification=1)
    assert fc.country_mask_[0] == False, "Country 0 should be filtered (diversification=1)"
    assert fc.n_countries_filtered_ >= 1, "At least one country should be filtered"
    
    # Check that country 0 has NaN fitness
    assert np.isnan(F[0]), "Filtered country should have NaN fitness"


def test_fitness_no_filtering():
    """Test that filtering can be disabled."""
    # Create matrix with unique product
    M = sp.lil_matrix((10, 10), dtype=float)
    M[0, 0] = 1  # Unique product
    
    # Fill rest
    for i in range(10):
        for j in range(1, 10):
            if np.random.rand() > 0.7:
                M[i, j] = 1
    
    M = M.tocsr()
    
    # Disable filtering
    fc = FitnessComplexity(min_ubiquity=1, min_diversification=1, verbose=False)
    F, Q = fc.fit_transform(M)
    
    # Check that nothing was filtered
    assert fc.n_countries_filtered_ == 0, "No countries should be filtered"
    assert fc.n_products_filtered_ == 0, "No products should be filtered"
    assert not np.any(np.isnan(F)), "No fitness values should be NaN"
    assert not np.any(np.isnan(Q)), "No complexity values should be NaN"


def test_fitness_iterative_filtering():
    """Test that iterative filtering removes cascading low-connectivity nodes."""
    # Create matrix where removing one node makes others low-connectivity
    M = sp.lil_matrix((10, 10), dtype=float)
    
    # Country 0 exports only product 0 (will be filtered)
    M[0, 0] = 1
    
    # Products 1-2 are only exported by countries 1-2 (diversification=2, borderline)
    M[1, 1] = 1
    M[1, 2] = 1
    M[2, 1] = 1
    M[2, 2] = 1
    
    # Rest have good connectivity
    for i in range(3, 10):
        for j in range(3, 10):
            if np.random.rand() > 0.5:
                M[i, j] = 1
    
    M = M.tocsr()
    
    # Run with iterative filtering
    fc = FitnessComplexity(
        min_ubiquity=3, 
        min_diversification=3, 
        iterative_filter=True,
        verbose=False
    )
    F, Q = fc.fit_transform(M)
    
    # With iterative filtering, low-connectivity nodes should be removed
    # even if they become low-connectivity only after first round
    n_filtered = fc.n_countries_filtered_ + fc.n_products_filtered_
    assert n_filtered > 0, "Iterative filtering should remove some nodes"


def test_fitness_atlas_2020_filtering():
    """Test that 2020 Atlas data gives reasonable results with filtering."""
    from fitkit import load_atlas_trade
    
    # Load 2020 data (known to have numerical issues without filtering)
    M, countries, products = load_atlas_trade(
        year=2020,
        classification='hs92',
        product_level=4,
        rca_threshold=1.0
    )
    
    # Run with default filtering
    fc = FitnessComplexity(verbose=False)
    F, Q = fc.fit_transform(M)
    
    # Check no single-country dominance
    F_valid = F[~np.isnan(F)]
    max_fitness = F_valid.max()
    median_fitness = np.median(F_valid)
    ratio = max_fitness / median_fitness
    
    assert ratio < 50, f"Max/median fitness ratio too high: {ratio:.1f} (should be < 50)"
    assert median_fitness > 0, f"Median fitness should be positive, got {median_fitness:.6f}"
    
    # Check that all valid countries have positive fitness (with proper filtering)
    n_nonzero = (F_valid > 1e-10).sum()
    n_total_valid = len(F_valid)
    fraction_nonzero = n_nonzero / n_total_valid
    
    assert fraction_nonzero > 0.95, (
        f"Too few countries with positive fitness: {n_nonzero}/{n_total_valid} "
        f"({100*fraction_nonzero:.1f}%)"
    )
    
    # Check that some filtering happened (2020 has 10 ubiquity=1 products)
    assert fc.n_products_filtered_ > 0, "Should filter some products in 2020 data"


def test_fitness_atlas_2000_vs_2020():
    """Compare 2000 vs 2020 to ensure both give reasonable results."""
    from fitkit import load_atlas_trade
    
    results = {}
    for year in [2000, 2020]:
        M, countries, products = load_atlas_trade(
            year=year,
            classification='hs92',
            product_level=4,
            rca_threshold=1.0
        )
        
        fc = FitnessComplexity(verbose=False)
        F, Q = fc.fit_transform(M)
        
        F_valid = F[~np.isnan(F)]
        results[year] = {
            'max_fitness': F_valid.max(),
            'median_fitness': np.median(F_valid),
            'n_nonzero': (F_valid > 1e-10).sum(),
            'n_total': len(F_valid),
            'n_filtered': fc.n_countries_filtered_ + fc.n_products_filtered_
        }
    
    # Both years should have reasonable ratios
    for year, res in results.items():
        ratio = res['max_fitness'] / res['median_fitness']
        assert ratio < 50, f"{year}: Max/median ratio too high: {ratio:.1f}"
        assert res['median_fitness'] > 0, f"{year}: Median fitness should be positive"
        
        fraction_nonzero = res['n_nonzero'] / res['n_total']
        assert fraction_nonzero > 0.95, (
            f"{year}: Too few countries with positive fitness: {fraction_nonzero:.1%}"
        )


def test_fitness_mask_dimensions():
    """Test that masks have correct dimensions."""
    M = sp.random(20, 30, density=0.15, format='csr')
    M.data[:] = 1  # Make binary
    
    fc = FitnessComplexity(min_ubiquity=2, min_diversification=3, verbose=False)
    F, Q = fc.fit_transform(M)
    
    # Check mask dimensions
    assert len(fc.country_mask_) == 20, "Country mask should match number of countries"
    assert len(fc.product_mask_) == 30, "Product mask should match number of products"
    
    # Check fitness/complexity dimensions
    assert len(F) == 20, "Fitness should match number of countries"
    assert len(Q) == 30, "Complexity should match number of products"
    
    # Check counts
    n_countries_kept = fc.country_mask_.sum()
    n_products_kept = fc.product_mask_.sum()
    
    assert fc.n_countries_filtered_ == 20 - n_countries_kept
    assert fc.n_products_filtered_ == 30 - n_products_kept


def test_fitness_filtering_empty_result():
    """Test handling when filtering removes everything."""
    # Create very sparse matrix
    M = sp.lil_matrix((10, 10), dtype=float)
    M[0, 0] = 1
    M[1, 1] = 1
    M = M.tocsr()
    
    # Aggressive filtering should remove almost everything
    fc = FitnessComplexity(min_ubiquity=5, min_diversification=5, verbose=False)
    
    # Should not crash, but will have mostly NaN results
    F, Q = fc.fit_transform(M)
    
    # Check that results are mostly NaN
    assert np.sum(~np.isnan(F)) < 5, "Most fitness values should be NaN after aggressive filtering"
    assert np.sum(~np.isnan(Q)) < 5, "Most complexity values should be NaN after aggressive filtering"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
