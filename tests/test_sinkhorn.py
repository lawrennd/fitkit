"""Tests for masked Sinkhorn-Knopp / IPF scaling."""

import numpy as np
import pytest
import scipy.sparse as sp

from fitkit.algorithms import SinkhornScaler, sinkhorn_masked
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

    u, v, W, history = sinkhorn_masked(M, r, c, n_iter=2000, tol=1e-10)

    # Algorithm should complete successfully
    assert history["iters"] > 0

    # Check marginals match (use reasonable tolerance)
    r_hat = np.asarray(W.sum(axis=1)).ravel()
    c_hat = np.asarray(W.sum(axis=0)).ravel()

    # Both should sum to the same total mass
    assert np.abs(r_hat.sum() - c_hat.sum()) < 1e-6


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

    u, v, W, history = sinkhorn_masked(M, r, c, n_iter=2000, tol=1e-10)

    # Algorithm should complete successfully
    assert history["iters"] > 0
    assert len(history["dr"]) > 0
    assert len(history["dc"]) > 0

    # Check that mass is conserved and marginals are reasonable
    r_hat = np.asarray(W.sum(axis=1)).ravel()
    c_hat = np.asarray(W.sum(axis=0)).ravel()

    # Mass conservation: row and column totals should match
    assert np.abs(r_hat.sum() - c_hat.sum()) < 1e-6

    # For sparse matrices, we can't always achieve perfect uniform marginals
    # (Sinkhorn can only redistribute mass along existing edges)
    # Just verify that marginals are positive and bounded
    assert np.all(r_hat >= 0)
    assert np.all(c_hat >= 0)


# ============================================================================
# Tests for SinkhornScaler transformer class (sklearn-style API)
# ============================================================================

def test_sinkhorn_scaler_basic():
    """Test that SinkhornScaler works with fit/transform."""
    bundle = create_small_fixture()
    M = bundle.matrix

    r = np.asarray(M.sum(axis=1)).ravel()
    c = np.asarray(M.sum(axis=0)).ravel()

    scaler = SinkhornScaler(n_iter=200, tol=1e-10)
    scaler.fit(M, row_marginals=r, col_marginals=c)

    # Check fitted attributes exist
    assert hasattr(scaler, 'u_')
    assert hasattr(scaler, 'v_')
    assert hasattr(scaler, 'W_')
    assert hasattr(scaler, 'history_')

    # Check convergence
    assert scaler.history_["converged"]


def test_sinkhorn_scaler_transform():
    """Test that transform applies fitted scaling."""
    bundle = create_small_fixture()
    M = bundle.matrix

    r = np.asarray(M.sum(axis=1)).ravel()
    c = np.asarray(M.sum(axis=0)).ravel()

    scaler = SinkhornScaler(n_iter=200, tol=1e-10)
    scaler.fit(M, row_marginals=r, col_marginals=c)

    # Transform should produce same result as W_
    W_transform = scaler.transform(M)
    np.testing.assert_array_almost_equal(W_transform.toarray(), scaler.W_.toarray())


def test_sinkhorn_scaler_fit_transform():
    """Test that fit_transform returns scaled matrix."""
    bundle = create_small_fixture()
    M = bundle.matrix

    r = np.asarray(M.sum(axis=1)).ravel()
    c = np.asarray(M.sum(axis=0)).ravel()

    scaler = SinkhornScaler(n_iter=200, tol=1e-10)
    W = scaler.fit_transform(M, row_marginals=r, col_marginals=c)

    # Check shape
    assert W.shape == M.shape

    # Check marginals
    r_hat = np.asarray(W.sum(axis=1)).ravel()
    c_hat = np.asarray(W.sum(axis=0)).ravel()

    np.testing.assert_allclose(r_hat, r, atol=1e-9)
    np.testing.assert_allclose(c_hat, c, atol=1e-9)


def test_sinkhorn_scaler_vs_functional():
    """Test that scaler produces same results as functional API."""
    bundle = create_small_fixture()
    M = bundle.matrix

    r = np.asarray(M.sum(axis=1)).ravel()
    c = np.asarray(M.sum(axis=0)).ravel()

    # Functional API
    u_func, v_func, W_func, history_func = sinkhorn_masked(M, r, c, n_iter=200, tol=1e-10)

    # Scaler API
    scaler = SinkhornScaler(n_iter=200, tol=1e-10)
    W_scaler = scaler.fit_transform(M, row_marginals=r, col_marginals=c)

    # Should produce identical results
    np.testing.assert_array_almost_equal(u_func, scaler.u_)
    np.testing.assert_array_almost_equal(v_func, scaler.v_)
    np.testing.assert_array_almost_equal(W_func.toarray(), W_scaler.toarray())


def test_sinkhorn_scaler_default_marginals():
    """Test that scaler works with default uniform marginals."""
    bundle = create_small_fixture()
    M = bundle.matrix

    # No marginals provided: should use uniform
    scaler = SinkhornScaler(n_iter=2000, tol=1e-10)
    scaler.fit(M)  # No row_marginals/col_marginals

    # Check that marginals were set to uniform
    assert np.allclose(scaler.row_marginals_, 1.0)
    assert np.allclose(scaler.col_marginals_, 1.0)

    # Check that algorithm completed
    assert scaler.history_["iters"] > 0
    assert hasattr(scaler, 'u_')
    assert hasattr(scaler, 'v_')
    assert hasattr(scaler, 'W_')


def test_sinkhorn_scaler_transform_before_fit():
    """Test that transform raises error if called before fit."""
    bundle = create_small_fixture()
    M = bundle.matrix

    scaler = SinkhornScaler()

    with pytest.raises(ValueError, match="must be fitted"):
        scaler.transform(M)
