"""Tests for ECI/PCI spectral computation."""

import numpy as np
import scipy.sparse as sp
import pytest

from fitkit.algorithms.eci import compute_eci_pci
from fitkit.data.fixtures import create_small_fixture


def test_eci_basic():
    """Test that ECI/PCI runs and returns correct shapes."""
    bundle = create_small_fixture()
    eci, pci = compute_eci_pci(bundle.matrix)
    
    # Check shapes
    assert eci.shape == (5,)  # 5 users
    assert pci.shape == (8,)  # 8 words


def test_eci_standardization():
    """Test that ECI/PCI are standardized (mean 0, std 1)."""
    bundle = create_small_fixture()
    eci, pci = compute_eci_pci(bundle.matrix)
    
    # Mean should be close to 0
    assert np.abs(eci.mean()) < 1e-10
    assert np.abs(pci.mean()) < 1e-10
    
    # Std should be close to 1
    assert np.abs(eci.std(ddof=0) - 1.0) < 1e-10
    assert np.abs(pci.std(ddof=0) - 1.0) < 1e-10


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
    """Test that ECI handles isolated nodes by dropping them."""
    # Create matrix with isolated node
    data = [1.0, 1.0, 1.0, 1.0]
    row = [0, 0, 1, 2]  # User 3 is isolated (no edges)
    col = [0, 1, 0, 1]
    M = sp.csr_matrix((data, (row, col)), shape=(4, 3), dtype=np.float64)
    
    # Isolated nodes are dropped internally, so shapes reflect connected component
    eci, pci = compute_eci_pci(M)
    
    # Check that computation completes without error
    assert eci.shape[0] == 3  # 3 connected users (user 3 dropped)
    assert pci.shape[0] == 2  # 2 connected words


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
    
    # In a fully connected graph, all nodes should have equal ECI/PCI (near zero after standardization)
    assert np.allclose(eci, 0.0, atol=1e-6)
    assert np.allclose(pci, 0.0, atol=1e-6)
