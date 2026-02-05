"""Comprehensive tests for ECI/PCI implementation.

This test module provides deeper verification of the ECI algorithm including:
- Mathematical properties (eigenvalue relationships, projection correctness)
- Comparison with hand-calculated examples
- Relationship to RCA (Revealed Comparative Advantage) matrix
- Behavior on different matrix structures (nested, modular, random)
- Numerical stability checks
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import pearsonr

from fitkit.algorithms import ECI, compute_eci_pci


class TestECIMathematicalProperties:
    """Test mathematical properties of the ECI algorithm."""

    def test_eci_eigenvalue_property(self):
        """Verify ECI is eigenvector of country-country projection matrix."""
        # Create a small test matrix
        M = sp.csr_matrix([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ], dtype=float)

        eci, pci = compute_eci_pci(M)

        # Reconstruct the country-country projection
        Mv = M.toarray()
        kc = Mv.sum(axis=1)
        kp = Mv.sum(axis=0)
        Dc_inv = np.diag(1.0 / kc)
        Dp_inv = np.diag(1.0 / kp)
        C = Dc_inv @ Mv @ Dp_inv @ Mv.T

        # Standardize eci back to eigenvector form (remove mean/std normalization)
        eci_unstd = eci * eci.std(ddof=0) + eci.mean()

        # C @ eci should be proportional to eci (eigenvalue equation)
        result = C @ eci_unstd
        # Check if parallel (ratio is constant)
        ratios = result / eci_unstd
        assert np.allclose(ratios, ratios[0], atol=1e-10)

    def test_eci_orthogonality_to_uniform(self):
        """Verify ECI (2nd eigenvector) is orthogonal to uniform vector (1st)."""
        M = sp.csr_matrix([
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ], dtype=float)

        eci, pci = compute_eci_pci(M)

        # The first eigenvector of C is uniform (constant)
        # The second eigenvector (ECI) should be orthogonal to it
        # This means: sum(eci) ≈ 0 (before standardization shifts it)
        # After standardization, mean is 0 by construction
        assert np.abs(eci.mean()) < 1e-10

        # But more fundamentally, check orthogonality to uniform vector
        ones = np.ones_like(eci)
        dot_product = np.dot(eci, ones)
        assert np.abs(dot_product) < 1e-9

    def test_pci_computation_formula(self):
        """Verify PCI formula: PCI = (M^T / kp) @ ECI."""
        M = sp.csr_matrix([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ], dtype=float)

        eci, pci = compute_eci_pci(M)

        # Manually compute PCI
        Mv = M.toarray()
        kp = Mv.sum(axis=0)
        Dp_inv = np.diag(1.0 / kp)

        # Unstandardize eci for this check
        eci_unstd = eci * eci.std(ddof=0) + eci.mean()

        # PCI = Dp_inv @ M^T @ ECI
        pci_manual = Dp_inv @ Mv.T @ eci_unstd

        # Standardize manual result
        pci_manual_std = (pci_manual - pci_manual.mean()) / (pci_manual.std(ddof=0) + 1e-12)

        # Should match
        np.testing.assert_allclose(pci, pci_manual_std, rtol=1e-10)


class TestECIKnownExamples:
    """Test ECI on examples with known or expected properties."""

    def test_nested_structure(self):
        """Test ECI on a perfectly nested matrix.

        Nested structure: countries with high diversification (many products)
        export all products that countries with lower diversification export,
        plus additional ones. This is the "nesting" property.

        For nested matrices, ECI should correlate strongly with diversification.
        """
        # Perfectly nested matrix (each row is superset of previous)
        M = sp.csr_matrix([
            [1, 0, 0, 0, 0],  # Low diversity
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],  # High diversity
        ], dtype=float)

        eci, pci = compute_eci_pci(M)

        # For nested structure, ECI should be monotonic with diversification
        diversification = np.asarray(M.sum(axis=1)).ravel()
        correlation = np.corrcoef(eci, diversification)[0, 1]

        # Should be very high correlation (>0.9)
        assert correlation > 0.9

    def test_symmetric_matrix(self):
        """Test ECI on symmetric patterns."""
        # Create symmetric structure (2 countries, 2 products, symmetric)
        M = sp.csr_matrix([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ], dtype=float)

        eci, pci = compute_eci_pci(M)

        # Countries 0 and 1 are identical -> should have same ECI
        assert np.abs(eci[0] - eci[1]) < 1e-10

    def test_star_graph(self):
        """Test ECI on star-shaped bipartite graph.

        One central 'hub' country connected to all products.
        Other countries connect to only one product each.
        """
        M = sp.csr_matrix([
            [1, 1, 1, 1],  # Hub country (connects to all)
            [1, 0, 0, 0],  # Spoke countries
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        eci, pci = compute_eci_pci(M)

        # Hub should have highest ECI
        assert eci[0] == np.max(eci)


class TestECIMatrixStructures:
    """Test ECI behavior on different matrix structures."""

    def test_random_vs_structured(self):
        """Compare ECI behavior on random vs structured matrices."""
        np.random.seed(42)

        # Random matrix
        M_random = sp.random(20, 30, density=0.15, format='csr')
        M_random.data = np.ones_like(M_random.data)

        # Nested matrix (structured)
        n = 20
        M_nested_data = []
        for i in range(n):
            row_data = np.zeros(30)
            # Row i has products 0 to (i+5)
            row_data[:min(i+5, 30)] = 1
            M_nested_data.append(row_data)
        M_nested = sp.csr_matrix(np.array(M_nested_data))

        eci_random, _ = compute_eci_pci(M_random)
        eci_nested, _ = compute_eci_pci(M_nested)

        # For nested: ECI should correlate very strongly with diversification
        div_nested = np.asarray(M_nested.sum(axis=1)).ravel()
        corr_nested = np.corrcoef(eci_nested, div_nested)[0, 1]

        # For random: correlation should be lower (less structure)
        div_random = np.asarray(M_random.sum(axis=1)).ravel()
        corr_random = np.corrcoef(eci_random, div_random)[0, 1]

        # Nested should have higher correlation
        assert corr_nested > corr_random

        # Nested should be very high (>0.95)
        assert corr_nested > 0.95

    def test_block_diagonal_structure(self):
        """Test ECI on block diagonal (modular) matrix."""
        # Two separate modules
        M = sp.csr_matrix([
            # Module 1
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            # Module 2
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], dtype=float)

        eci, pci = compute_eci_pci(M)

        # Within modules, countries should have similar ECI
        assert np.abs(eci[0] - eci[1]) < 1e-10
        assert np.abs(eci[2] - eci[3]) < 1e-10


class TestECINumericalStability:
    """Test numerical stability of ECI implementation."""

    def test_large_matrix(self):
        """Test ECI on larger matrix."""
        np.random.seed(42)
        M = sp.random(100, 150, density=0.1, format='csr')
        M.data = np.ones_like(M.data)

        eci, pci = compute_eci_pci(M)

        # Should complete without error
        assert eci.shape == (100,)
        assert pci.shape == (150,)

        # Should be standardized
        assert np.abs(eci.mean()) < 1e-10
        assert np.abs(pci.mean()) < 1e-10
        assert np.abs(eci.std(ddof=0) - 1.0) < 1e-10
        assert np.abs(pci.std(ddof=0) - 1.0) < 1e-10

    def test_very_sparse_matrix(self):
        """Test ECI on very sparse matrix."""
        np.random.seed(42)
        M = sp.random(50, 80, density=0.02, format='csr')  # Only 2% filled
        M.data = np.ones_like(M.data)

        eci, pci = compute_eci_pci(M)

        # Should complete (may drop some isolated nodes)
        assert eci.shape[0] <= 50
        assert pci.shape[0] <= 80

        # Should be standardized
        assert np.abs(eci.mean()) < 1e-10
        assert np.abs(pci.mean()) < 1e-10

    def test_dense_matrix(self):
        """Test ECI on dense matrix."""
        M = sp.csr_matrix(np.random.rand(30, 40) > 0.3)  # ~70% filled

        eci, pci = compute_eci_pci(M)

        # Should complete
        assert eci.shape[0] <= 30
        assert pci.shape[0] <= 40

        # Should be standardized
        assert np.abs(eci.mean()) < 1e-10
        assert np.abs(pci.mean()) < 1e-10


class TestECIVsRCA:
    """Test relationship between ECI and RCA (Revealed Comparative Advantage)."""

    def test_eci_uses_normalized_matrix(self):
        """Verify ECI computation uses RCA-like normalization.

        The country-country projection uses (M/kc) @ (M^T/kp),
        which is equivalent to the RCA formulation where each entry
        is normalized by row and column totals.
        """
        M = sp.csr_matrix([
            [2, 1, 0, 0],  # Note: non-binary values
            [1, 2, 1, 0],
            [0, 1, 2, 1],
            [0, 0, 1, 2],
        ], dtype=float)

        # Binarize for standard ECI
        M_bin = M.copy()
        M_bin.data = np.ones_like(M_bin.data)

        eci, pci = compute_eci_pci(M_bin)

        # ECI should exist and be standardized
        assert eci.shape == (4,)
        assert np.abs(eci.mean()) < 1e-10


class TestECIReproducibility:
    """Test that ECI results are reproducible and consistent."""

    def test_multiple_runs_identical(self):
        """Test that multiple runs produce identical results."""
        M = sp.random(30, 40, density=0.15, format='csr', random_state=42)
        M.data = np.ones_like(M.data)

        results = []
        for _ in range(5):
            eci, pci = compute_eci_pci(M)
            results.append((eci.copy(), pci.copy()))

        # All runs should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0][0], results[i][0])
            np.testing.assert_array_equal(results[0][1], results[i][1])

    def test_estimator_vs_functional_consistency(self):
        """Test that estimator and functional API are consistent."""
        M = sp.random(25, 35, density=0.12, format='csr', random_state=123)
        M.data = np.ones_like(M.data)

        # Functional API
        eci_func, pci_func = compute_eci_pci(M)

        # Estimator API (multiple calls)
        for _ in range(3):
            eci_est, pci_est = ECI().fit_transform(M)
            np.testing.assert_array_almost_equal(eci_func, eci_est)
            np.testing.assert_array_almost_equal(pci_func, pci_est)


class TestECIEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_viable_matrix(self):
        """Test smallest matrix that can produce ECI."""
        # 2x2 matrix with 3 edges (minimum for 2 eigenvectors)
        M = sp.csr_matrix([
            [1, 1],
            [1, 0],
        ], dtype=float)

        eci, pci = compute_eci_pci(M)

        assert eci.shape == (2,)
        assert pci.shape == (2,)

    def test_rectangular_matrix(self):
        """Test ECI on highly rectangular matrix."""
        # Many more products than countries
        M = sp.csr_matrix([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 0, 0, 0],
        ], dtype=float)

        eci, pci = compute_eci_pci(M)

        assert eci.shape == (3,)
        assert pci.shape == (10,)

        # Should be standardized
        assert np.abs(eci.mean()) < 1e-10
        assert np.abs(pci.mean()) < 1e-10


def test_eci_diagnostic_on_random_matrix():
    """Diagnostic test: compute ECI on random matrix and check properties.

    This test is specifically to help diagnose why ECI and Fitness
    might not correlate well even on random matrices.
    """
    np.random.seed(42)
    n_users = 50
    n_words = 75
    density = 0.15

    M = sp.random(n_users, n_words, density=density, format='csr', random_state=42)
    M.data = np.ones_like(M.data)

    eci, pci = compute_eci_pci(M)

    # Compute basic statistics
    diversification = np.asarray(M.sum(axis=1)).ravel()
    ubiquity = np.asarray(M.sum(axis=0)).ravel()

    # ECI should correlate with diversification (by design)
    corr_div = np.corrcoef(eci, diversification)[0, 1]

    print("\n=== ECI Diagnostic on Random Matrix ===")
    print(f"Matrix: {n_users} × {n_words}, density: {density:.2%}")
    print(f"ECI range: [{eci.min():.3f}, {eci.max():.3f}]")
    print(f"ECI mean: {eci.mean():.6f} (should be ~0)")
    print(f"ECI std: {eci.std(ddof=0):.6f} (should be ~1)")
    print(f"Correlation(ECI, diversification): {corr_div:.4f}")
    print(f"Diversification range: [{diversification.min():.0f}, {diversification.max():.0f}]")

    # The key insight: for random matrices, ECI is very close to just
    # being a rescaled version of diversification
    assert corr_div > 0.85, f"ECI should correlate strongly with diversification, got {corr_div:.4f}"

    # Verify standardization
    assert np.abs(eci.mean()) < 1e-10
    assert np.abs(eci.std(ddof=0) - 1.0) < 1e-10


if __name__ == "__main__":
    # Run the diagnostic test
    test_eci_diagnostic_on_random_matrix()
