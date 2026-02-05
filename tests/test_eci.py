"""Tests for ECI/PCI spectral computation."""

import numpy as np
import pytest
import scipy.sparse as sp

from fitkit.algorithms import ECI, compute_eci_pci
from fitkit.data.fixtures import create_small_fixture


def test_eci_basic():
    """Test that ECI/PCI runs and returns correct shapes."""
    bundle = create_small_fixture()
    eci, pci = compute_eci_pci(bundle.matrix)

    # Check shapes
    assert eci.shape == (5,)  # 5 users
    assert pci.shape == (8,)  # 8 words


def test_eci_standardization():
    """Test that ECI/PCI are standardized (mean 0, std 1).
    
    Note: Standardization is computed on connected nodes only (ignoring NaN).
    """
    bundle = create_small_fixture()
    eci, pci = compute_eci_pci(bundle.matrix)

    # Remove any NaN values before computing statistics
    eci_valid = eci[~np.isnan(eci)]
    pci_valid = pci[~np.isnan(pci)]
    
    # Mean should be close to 0
    assert np.abs(eci_valid.mean()) < 1e-10
    assert np.abs(pci_valid.mean()) < 1e-10

    # Std should be close to 1
    assert np.abs(eci_valid.std(ddof=0) - 1.0) < 1e-10
    assert np.abs(pci_valid.std(ddof=0) - 1.0) < 1e-10


def test_eci_sign_convention():
    """Test that ECI correlates positively with diversification."""
    bundle = create_small_fixture()
    M = bundle.matrix
    eci, pci = compute_eci_pci(M)

    # Compute diversification (row sums)
    diversification = np.asarray(M.sum(axis=1)).ravel()

    # ECI should correlate positively with diversification
    correlation = np.corrcoef(eci, diversification)[0, 1]
    assert correlation > 0


def test_eci_deterministic():
    """Test that ECI/PCI is deterministic (same input → same output)."""
    bundle = create_small_fixture()

    eci1, pci1 = compute_eci_pci(bundle.matrix)
    eci2, pci2 = compute_eci_pci(bundle.matrix)

    np.testing.assert_array_almost_equal(eci1, eci2)
    np.testing.assert_array_almost_equal(pci1, pci2)


def test_eci_ranking():
    """Test that ECI rankings are stable."""
    bundle = create_small_fixture()

    eci, pci = compute_eci_pci(bundle.matrix)

    # Get rankings
    user_ranks = eci.argsort()
    word_ranks = pci.argsort()

    # Re-run and check rankings are identical
    eci2, pci2 = compute_eci_pci(bundle.matrix)
    user_ranks2 = eci2.argsort()
    word_ranks2 = pci2.argsort()

    np.testing.assert_array_equal(user_ranks, user_ranks2)
    np.testing.assert_array_equal(word_ranks, word_ranks2)


def test_eci_isolated_nodes():
    """Test that ECI handles isolated nodes by setting them to NaN."""
    # Create matrix with isolated node
    data = [1.0, 1.0, 1.0, 1.0]
    row = [0, 0, 1, 2]  # User 3 is isolated (no edges)
    col = [0, 1, 0, 1]
    M = sp.csr_matrix((data, (row, col)), shape=(4, 3), dtype=np.float64)

    # Should issue a warning about isolated nodes
    with pytest.warns(UserWarning, match="Dropped 1 isolated countries"):
        eci, pci = compute_eci_pci(M)

    # Output dimensions should match input
    assert eci.shape[0] == 4  # All 4 users in output
    assert pci.shape[0] == 3  # All 3 products in output (product 2 is isolated)
    
    # Isolated nodes should be NaN
    assert np.isnan(eci[3])  # User 3 is isolated
    assert np.isnan(pci[2])  # Product 2 is isolated
    
    # Connected nodes should have values
    assert not np.isnan(eci[0])
    assert not np.isnan(eci[1])
    assert not np.isnan(eci[2])
    assert not np.isnan(pci[0])
    assert not np.isnan(pci[1])


def test_eci_insufficient_structure():
    """Test that ECI raises error for insufficient structure."""
    # Single edge: not enough for eigenvector computation
    M = sp.csr_matrix(([1.0], ([0], [0])), shape=(2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="Not enough dimensions"):
        compute_eci_pci(M)


def test_eci_fully_connected():
    """Test ECI on a fully connected bipartite graph."""
    # All users connect to all words
    M = sp.csr_matrix(np.ones((4, 3)), dtype=np.float64)

    eci, pci = compute_eci_pci(M)

    # Check shapes
    assert eci.shape == (4,)
    assert pci.shape == (3,)

    # Check standardization (mean 0, std 1)
    assert np.abs(eci.mean()) < 1e-10
    assert np.abs(pci.mean()) < 1e-10

    # Note: Fully connected doesn't mean uniform ECI/PCI
    # The second eigenvector can still have structure depending on dimensions


# ============================================================================
# Tests for ECI estimator class (sklearn-style API)
# ============================================================================

def test_eci_estimator_basic():
    """Test that ECI estimator works with fit/transform."""
    bundle = create_small_fixture()

    eci_est = ECI()
    eci_est.fit(bundle.matrix)

    # Check fitted attributes exist
    assert hasattr(eci_est, 'eci_')
    assert hasattr(eci_est, 'pci_')

    # Check shapes
    assert eci_est.eci_.shape == (5,)
    assert eci_est.pci_.shape == (8,)


def test_eci_estimator_fit_transform():
    """Test that fit_transform returns (ECI, PCI)."""
    bundle = create_small_fixture()

    eci_est = ECI()
    eci, pci = eci_est.fit_transform(bundle.matrix)

    # Check shapes
    assert eci.shape == (5,)
    assert pci.shape == (8,)

    # Check that fitted attributes match
    np.testing.assert_array_equal(eci, eci_est.eci_)
    np.testing.assert_array_equal(pci, eci_est.pci_)


def test_eci_estimator_vs_functional():
    """Test that estimator produces same results as functional API."""
    bundle = create_small_fixture()

    # Functional API
    eci_func, pci_func = compute_eci_pci(bundle.matrix)

    # Estimator API
    eci_est, pci_est = ECI().fit_transform(bundle.matrix)

    # Should produce identical results
    np.testing.assert_array_almost_equal(eci_func, eci_est)
    np.testing.assert_array_almost_equal(pci_func, pci_est)


def test_eci_estimator_chaining():
    """Test that estimator can be used in method chaining."""
    bundle = create_small_fixture()

    # Chain: instantiate -> fit -> access attributes
    eci = ECI().fit(bundle.matrix).eci_

    assert eci.shape == (5,)
    # Check standardization
    assert np.abs(eci.mean()) < 1e-10
