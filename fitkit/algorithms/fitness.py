"""Fitness-Complexity fixed-point iteration.

This module implements the nonlinear Fitness-Complexity algorithm
as described in:

    Lawrence, N.D. (2024). "Conditional Likelihood Interpretation of
    Economic Fitness" (working paper).

The algorithm computes country fitness F and product complexity Q
via alternating harmonic aggregation on the bipartite support graph.

Provides both a scikit-learn-style estimator (FitnessComplexity) and a
functional API (fitness_complexity) for convenience and backward compatibility.
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional


def fitness_complexity(
    M_bin: sp.spmatrix,
    n_iter: int = 200,
    tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute Fitness-Complexity fixed point on binary incidence matrix M.

    This function iterates the Fitness-Complexity updates:
        F_u = sum_p M_{up} Q_p  (diversification-weighted complexity)
        Q_p = 1 / sum_u M_{up} / F_u  (harmonic mean of fitness)

    Both F and Q are normalized to unit mean at each iteration.

    Args:
        M_bin: Scipy sparse matrix (n_rows × n_cols), entries in {0,1}.
               Rows represent countries/users, columns represent products/words.
        n_iter: Maximum number of iterations (default: 200).
        tol: Convergence tolerance on max absolute change (default: 1e-10).

    Returns:
        F: Country/user fitness scores (n_rows,), normalized to mean 1.
        Q: Product/word complexity scores (n_cols,), normalized to mean 1.
        history: Dict with convergence diagnostics:
            - "dF": list of max absolute changes in F per iteration
            - "dQ": list of max absolute changes in Q per iteration

    Notes:
        - The algorithm is gauge-invariant (scale-free): multiplying F by a
          constant divides Q by the same constant, leaving rankings unchanged.
        - Convergence is typically fast (tens of iterations) for well-connected graphs.
        - The fixed point is unique up to the scale gauge when the support graph
          is connected.

    References:
        Lawrence, N.D. (2024). "Conditional Likelihood Interpretation of Economic Fitness".
        Tacchella et al. (2012). "A New Metrics for Countries' Fitness and Products' Complexity".
    """
    n_rows, n_cols = M_bin.shape
    F = np.ones(n_rows, dtype=float)
    Q = np.ones(n_cols, dtype=float)

    M_csr = M_bin.tocsr()

    history = {"dF": [], "dQ": []}

    for it in range(n_iter):
        # Update F: diversification-weighted complexity
        F_new = M_csr @ Q
        F_new = np.maximum(F_new, 1e-12)  # Numerical guard
        F_new = F_new / F_new.mean()  # Normalize to unit mean

        # Update Q: harmonic mean of fitness
        invF = 1.0 / F_new
        denom = M_csr.T @ invF  # denom_p = sum_u M_{up}/F_u
        denom = np.maximum(denom, 1e-12)  # Numerical guard
        Q_new = 1.0 / denom
        Q_new = Q_new / Q_new.mean()  # Normalize to unit mean

        # Track convergence
        dF = float(np.max(np.abs(F_new - F)))
        dQ = float(np.max(np.abs(Q_new - Q)))
        history["dF"].append(dF)
        history["dQ"].append(dQ)

        F, Q = F_new, Q_new

        # Check convergence
        if max(dF, dQ) < tol:
            print(f"Converged in {it+1} iterations")
            break

    return F, Q, history


class FitnessComplexity:
    """Scikit-learn-style estimator for Fitness-Complexity fixed-point iteration.
    
    This estimator computes country/user fitness and product/word complexity
    via the nonlinear Fitness-Complexity fixed-point iteration. It follows
    scikit-learn conventions: hyperparameters in __init__, data in fit().
    
    Parameters:
        n_iter: Maximum number of iterations (default: 200).
        tol: Convergence tolerance on max absolute change (default: 1e-10).
        verbose: If True, print convergence message (default: True).
    
    Attributes (set after calling fit):
        fitness_: Fitted country/user fitness scores (n_rows,), normalized to mean 1.
        complexity_: Fitted product/word complexity scores (n_cols,), normalized to mean 1.
        history_: Dict with convergence diagnostics (dF, dQ lists).
        n_iter_: Number of iterations performed.
    
    Examples:
        >>> from fitkit.algorithms import FitnessComplexity
        >>> fc = FitnessComplexity(n_iter=200, tol=1e-10)
        >>> fc.fit(M)  # M is binary incidence matrix
        >>> F = fc.fitness_
        >>> Q = fc.complexity_
        
        >>> # Or: one-liner
        >>> F, Q = FitnessComplexity(n_iter=200).fit_transform(M)
    
    References:
        Lawrence, N.D. (2024). "Conditional Likelihood Interpretation of Economic Fitness".
        Tacchella et al. (2012). "A New Metrics for Countries' Fitness and Products' Complexity".
    """
    
    def __init__(self, n_iter: int = 200, tol: float = 1e-10, verbose: bool = True):
        """Initialize FitnessComplexity estimator.
        
        Args:
            n_iter: Maximum number of iterations.
            tol: Convergence tolerance.
            verbose: If True, print convergence message.
        """
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
    
    def fit(self, X: sp.spmatrix, y: Optional[np.ndarray] = None):
        """Compute Fitness-Complexity fixed point on binary incidence matrix X.
        
        Args:
            X: Scipy sparse matrix (n_rows × n_cols), entries in {0,1}.
               Rows represent countries/users, columns represent products/words.
            y: Ignored. Present for sklearn compatibility.
        
        Returns:
            self: Fitted estimator.
        """
        # Temporarily suppress print for non-verbose mode
        import io
        import sys
        
        if not self.verbose:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
        
        try:
            self.fitness_, self.complexity_, self.history_ = fitness_complexity(
                X, n_iter=self.n_iter, tol=self.tol
            )
            self.n_iter_ = len(self.history_["dF"])
        finally:
            if not self.verbose:
                sys.stdout = old_stdout
        
        return self
    
    def fit_transform(self, X: sp.spmatrix, y: Optional[np.ndarray] = None):
        """Fit and return (fitness, complexity).
        
        Args:
            X: Scipy sparse matrix (n_rows × n_cols), entries in {0,1}.
            y: Ignored. Present for sklearn compatibility.
        
        Returns:
            fitness: Country/user fitness scores (n_rows,).
            complexity: Product/word complexity scores (n_cols,).
        """
        self.fit(X, y)
        return self.fitness_, self.complexity_
