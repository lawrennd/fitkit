"""Tests for Fitness-Complexity fixed-point iteration."""

import numpy as np
import scipy.sparse as sp

from fitkit.algorithms import FitnessComplexity, fitness_complexity
from fitkit.data.fixtures import create_small_fixture


def test_fitness_convergence():
    """Test that fitness-complexity converges on a small fixture."""
    bundle = create_small_fixture()
    F, Q, history = fitness_complexity(bundle.matrix, n_iter=100, tol=1e-10)

    # Check shapes
    assert F.shape == (5,)  # 5 users
    assert Q.shape == (8,)  # 8 words

    # Check convergence (use reasonable tolerance for small fixture)
    assert len(history["dF"]) > 0
    assert len(history["dQ"]) > 0
    # Convergence should improve over iterations (last < first)
    assert history["dF"][-1] < history["dF"][0]
    assert history["dQ"][-1] < history["dQ"][0]

    # Check gauge: mean should be 1.0
    assert np.abs(F.mean() - 1.0) < 1e-10
    assert np.abs(Q.mean() - 1.0) < 1e-10


def test_fitness_positivity():
    """Test that fitness and complexity are always positive."""
    bundle = create_small_fixture()
    F, Q, _ = fitness_complexity(bundle.matrix, n_iter=100)

    assert np.all(F > 0)
    assert np.all(Q > 0)


def test_fitness_isolated_node():
    """Test that fitness handles isolated nodes gracefully."""
    # Create matrix with an isolated user (row of zeros)
    data = [1.0, 1.0, 1.0, 1.0, 1.0]
    row = [0, 0, 1, 1, 2]  # User 2 connects, user 3 is isolated
    col = [0, 1, 0, 2, 1]
    M = sp.csr_matrix((data, (row, col)), shape=(4, 3), dtype=np.float64)

    F, Q, _ = fitness_complexity(M, n_iter=50)

    # Isolated user (row 3) should have very low fitness
    assert F[3] < F[:3].min()

    # Check shapes
    assert F.shape == (4,)
    assert Q.shape == (3,)


def test_fitness_deterministic():
    """Test that fitness-complexity is deterministic (same input → same output)."""
    bundle = create_small_fixture()

    F1, Q1, _ = fitness_complexity(bundle.matrix, n_iter=100, tol=1e-10)
    F2, Q2, _ = fitness_complexity(bundle.matrix, n_iter=100, tol=1e-10)

    np.testing.assert_array_almost_equal(F1, F2)
    np.testing.assert_array_almost_equal(Q1, Q2)


def test_fitness_ranking_stability():
    """Test that fitness rankings are stable across runs."""
    bundle = create_small_fixture()

    F, Q, _ = fitness_complexity(bundle.matrix, n_iter=100)

    # Get rankings
    user_ranks = F.argsort()[::-1]  # Descending order
    word_ranks = Q.argsort()[::-1]

    # Re-run and check rankings are identical
    F2, Q2, _ = fitness_complexity(bundle.matrix, n_iter=100)
    user_ranks2 = F2.argsort()[::-1]
    word_ranks2 = Q2.argsort()[::-1]

    np.testing.assert_array_equal(user_ranks, user_ranks2)
    np.testing.assert_array_equal(word_ranks, word_ranks2)


def test_fitness_scale_invariance():
    """Test that fitness is invariant to matrix scaling (gauge freedom)."""
    bundle = create_small_fixture()
    M = bundle.matrix

    # Compute fitness on original matrix
    F1, Q1, _ = fitness_complexity(M, n_iter=100)
    ranks_F1 = F1.argsort()
    ranks_Q1 = Q1.argsort()

    # Scale matrix by constant (shouldn't affect binary incidence, but test anyway)
    M_scaled = M * 2.0
    M_scaled.data = np.ones_like(M_scaled.data)  # Re-binarize

    F2, Q2, _ = fitness_complexity(M_scaled, n_iter=100)
    ranks_F2 = F2.argsort()
    ranks_Q2 = Q2.argsort()

    # Rankings should be identical
    np.testing.assert_array_equal(ranks_F1, ranks_F2)
    np.testing.assert_array_equal(ranks_Q1, ranks_Q2)


def test_fitness_empty_matrix():
    """Test that fitness handles empty matrices gracefully."""
    M = sp.csr_matrix((5, 8), dtype=np.float64)  # All zeros

    # Should complete without crashing (though results are degenerate)
    F, Q, _ = fitness_complexity(M, n_iter=10)

    assert F.shape == (5,)
    assert Q.shape == (8,)
    # All values should be equal (uniform) for empty matrix
    assert np.allclose(F, F.mean())
    assert np.allclose(Q, Q.mean())


# ============================================================================
# Tests for FitnessComplexity estimator class (sklearn-style API)
# ============================================================================

def test_fitness_estimator_basic():
    """Test that FitnessComplexity estimator works with fit/transform."""
    bundle = create_small_fixture()

    fc = FitnessComplexity(n_iter=100, tol=1e-10, verbose=False)
    fc.fit(bundle.matrix)

    # Check fitted attributes exist
    assert hasattr(fc, 'fitness_')
    assert hasattr(fc, 'complexity_')
    assert hasattr(fc, 'history_')
    assert hasattr(fc, 'n_iter_')

    # Check shapes
    assert fc.fitness_.shape == (5,)
    assert fc.complexity_.shape == (8,)


def test_fitness_estimator_fit_transform():
    """Test that fit_transform returns (F, Q)."""
    bundle = create_small_fixture()

    fc = FitnessComplexity(n_iter=100, tol=1e-10, verbose=False)
    F, Q = fc.fit_transform(bundle.matrix)

    # Check shapes
    assert F.shape == (5,)
    assert Q.shape == (8,)

    # Check that fitted attributes match
    np.testing.assert_array_equal(F, fc.fitness_)
    np.testing.assert_array_equal(Q, fc.complexity_)


def test_fitness_estimator_vs_functional():
    """Test that estimator produces same results as functional API."""
    bundle = create_small_fixture()

    # Functional API
    F_func, Q_func, history_func = fitness_complexity(bundle.matrix, n_iter=100, tol=1e-10)

    # Estimator API
    fc = FitnessComplexity(n_iter=100, tol=1e-10, verbose=False)
    F_est, Q_est = fc.fit_transform(bundle.matrix)

    # Should produce identical results
    np.testing.assert_array_almost_equal(F_func, F_est)
    np.testing.assert_array_almost_equal(Q_func, Q_est)


def test_fitness_estimator_chaining():
    """Test that estimator can be used in method chaining."""
    bundle = create_small_fixture()

    # Chain: instantiate -> fit -> access attributes
    F = FitnessComplexity(n_iter=100, verbose=False).fit(bundle.matrix).fitness_

    assert F.shape == (5,)
    assert np.all(F > 0)


def test_fitness_estimator_parameters():
    """Test that estimator parameters are stored correctly."""
    fc = FitnessComplexity(n_iter=500, tol=1e-12, verbose=True)

    assert fc.n_iter == 500
    assert fc.tol == 1e-12
    assert fc.verbose
