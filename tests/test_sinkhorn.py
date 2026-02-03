"""Tests for masked Sinkhorn-Knopp / IPF scaling."""

import numpy as np
import scipy.sparse as sp
import pytest

from fitkit.algorithms.sinkhorn import sinkhorn_masked
from fitkit.data.fixtures import create_small_fixture


def test_sinkhorn_convergence():
    """Test that Sinkhorn converges on a small fixture."""
    bundle = create_small_fixture()
    M = bundle.matrix
    
    # Desired marginals (proportional to current margins)
    r = np.asarray(M.sum(axis=1)).ravel()
    c = np.asarray(M.sum(axis=0)).ravel()
    
    u, v, W, history = sinkhorn_masked(M, r, c, n_iter=200, tol=1e-10)
    
    # Check convergence
    assert history["converged"]
    assert history["iters"] < 200  # Should converge quickly


def test_sinkhorn_marginals():
    """Test that Sinkhorn matches desired marginals."""
    bundle = create_small_fixture()
    M = bundle.matrix
    
    # Desired marginals
    r = np.asarray(M.sum(axis=1)).ravel()
    c = np.asarray(M.sum(axis=0)).ravel()
    
    u, v, W, history = sinkhorn_masked(M, r, c, n_iter=200, tol=1e-10)
    
    # Check marginals
    r_hat = np.asarray(W.sum(axis=1)).ravel()
    c_hat = np.asarray(W.sum(axis=0)).ravel()
    
    np.testing.assert_allclose(r_hat, r, atol=1e-9)
    np.testing.assert_allclose(c_hat, c, atol=1e-9)


def test_sinkhorn_support_preservation():
    """Test that Sinkhorn preserves support (sparsity pattern)."""
    bundle = create_small_fixture()
    M = bundle.matrix
    
    r = np.asarray(M.sum(axis=1)).ravel()
    c = np.asarray(M.sum(axis=0)).ravel()
    
    u, v, W, history = sinkhorn_masked(M, r, c, n_iter=200, tol=1e-10)
    
    # Check that W has the same support as M
    assert W.nnz == M.nnz  # Same number of nonzeros
    
    # Check that nonzero positions match
    M_coo = M.tocoo()
    W_coo = W.tocoo()
    
    M_coords = set(zip(M_coo.row, M_coo.col))
    W_coords = set(zip(W_coo.row, W_coo.col))
    
    assert M_coords == W_coords


def test_sinkhorn_deterministic():
    """Test that Sinkhorn is deterministic (same input → same output)."""
    bundle = create_small_fixture()
    M = bundle.matrix
    
    r = np.asarray(M.sum(axis=1)).ravel()
    c = np.asarray(M.sum(axis=0)).ravel()
    
    u1, v1, W1, _ = sinkhorn_masked(M, r, c, n_iter=200, tol=1e-10)
    u2, v2, W2, _ = sinkhorn_masked(M, r, c, n_iter=200, tol=1e-10)
    
    np.testing.assert_array_almost_equal(u1, u2)
    np.testing.assert_array_almost_equal(v1, v2)
    np.testing.assert_array_almost_equal(W1.toarray(), W2.toarray())


def test_sinkhorn_normalization():
    """Test that Sinkhorn automatically normalizes marginals to equal mass."""
    bundle = create_small_fixture()
    M = bundle.matrix
    
    # Unnormalized marginals
    r = np.ones(M.shape[0])  # Total mass: 5
    c = np.ones(M.shape[1]) * 2  # Total mass: 16
    
    u, v, W, history = sinkhorn_masked(M, r, c, n_iter=200, tol=1e-10)
    
    # Should still converge (c is auto-normalized internally)
    assert history["converged"]
    
    # Check marginals match after normalization
    r_hat = np.asarray(W.sum(axis=1)).ravel()
    c_hat = np.asarray(W.sum(axis=0)).ravel()
    
    # Both should sum to the same total mass
    assert np.abs(r_hat.sum() - c_hat.sum()) < 1e-9


def test_sinkhorn_infeasible_isolated_row():
    """Test that Sinkhorn raises error for isolated row with positive mass."""
    # Create matrix with isolated row
    data = [1.0, 1.0, 1.0]
    row = [0, 1, 1]  # Row 2 is isolated (no edges)
    col = [0, 0, 1]
    M = sp.csr_matrix((data, (row, col)), shape=(3, 2), dtype=np.float64)
    
    # Desired marginals: row 2 has positive mass (infeasible)
    r = np.array([1.0, 2.0, 1.0])  # Row 2 has mass 1.0 but no edges!
    c = np.array([3.0, 1.0])
    
    with pytest.raises(ValueError, match="Infeasible.*row with zero support"):
        sinkhorn_masked(M, r, c, n_iter=100)


def test_sinkhorn_infeasible_isolated_col():
    """Test that Sinkhorn raises error for isolated column with positive mass."""
    # Create matrix with isolated column
    data = [1.0, 1.0, 1.0]
    row = [0, 1, 1]
    col = [0, 0, 1]  # Column 2 is isolated (no edges)
    M = sp.csr_matrix((data, (row, col)), shape=(3, 3), dtype=np.float64)
    
    # Desired marginals: column 2 has positive mass (infeasible)
    r = np.array([1.0, 2.0, 0.0])
    c = np.array([2.0, 1.0, 0.5])  # Column 2 has mass 0.5 but no edges!
    
    with pytest.raises(ValueError, match="Infeasible.*col with zero support"):
        sinkhorn_masked(M, r, c, n_iter=100)


def test_sinkhorn_shape_mismatch():
    """Test that Sinkhorn raises error for shape mismatches."""
    M = sp.csr_matrix(np.ones((3, 4)), dtype=np.float64)
    
    # Wrong r shape
    r = np.ones(2)  # Should be 3
    c = np.ones(4)
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        sinkhorn_masked(M, r, c, n_iter=100)


def test_sinkhorn_uniform_marginals():
    """Test Sinkhorn with uniform desired marginals."""
    bundle = create_small_fixture()
    M = bundle.matrix
    
    # Uniform marginals
    r = np.ones(M.shape[0])
    c = np.ones(M.shape[1])
    
    u, v, W, history = sinkhorn_masked(M, r, c, n_iter=200, tol=1e-10)
    
    # Should converge
    assert history["converged"]
    
    # Check marginals
    r_hat = np.asarray(W.sum(axis=1)).ravel()
    c_hat = np.asarray(W.sum(axis=0)).ravel()
    
    # Should match uniform distribution (up to total mass)
    assert np.allclose(r_hat / r_hat.sum(), r / r.sum(), atol=1e-9)
    assert np.allclose(c_hat / c_hat.sum(), c / c.sum(), atol=1e-9)
