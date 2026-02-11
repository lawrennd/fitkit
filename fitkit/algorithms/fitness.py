"""Fitness-Complexity fixed-point iteration.

This module implements the nonlinear Fitness-Complexity algorithm
as described in:

    Lawrence, N.D. (2024). "Conditional Likelihood Interpretation of
    Economic Fitness" (working paper).

The algorithm computes country fitness F and product complexity Q
via alternating harmonic aggregation on the bipartite support graph.

Provides a scikit-learn-style estimator (FitnessComplexity) as the primary API,
with a deprecated functional API (fitness_complexity) for backward compatibility.
"""

import warnings
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Dict


def _fitness_complexity(
    M_bin: sp.spmatrix,
    n_iter: int = 200,
    tol: float = 1e-10,
    return_history: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict]:
    """Compute Fitness-Complexity fixed point (private function).

    This function iterates the Fitness-Complexity updates:
        F_u = sum_p M_{up} Q_p  (diversification-weighted complexity)
        Q_p = 1 / sum_u M_{up} / F_u  (harmonic mean of fitness)

    Both F and Q are normalized to unit mean at each iteration.
    
    Note:
        This is a private function. Use FitnessComplexity class instead.

    Args:
        M_bin: Scipy sparse matrix (n_rows × n_cols), entries in {0,1}.
               Rows represent countries/users, columns represent products/words.
        n_iter: Maximum number of iterations (default: 200).
        tol: Convergence tolerance on max absolute change (default: 1e-10).
        return_history: If True, return detailed convergence history.
        verbose: If True, print convergence message.

    Returns:
        F: Country/user fitness scores (n_rows,), normalized to mean 1.
        Q: Product/word complexity scores (n_cols,), normalized to mean 1.
        history (optional): Dict with convergence diagnostics

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

    history: Dict = {"dF": [], "dQ": []} if return_history else {}
    converged = False

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
        if return_history:
            history["dF"].append(dF)
            history["dQ"].append(dQ)

        F, Q = F_new, Q_new

        # Check convergence
        if max(dF, dQ) < tol:
            converged = True
            if verbose:
                print(f"Converged in {it+1} iterations")
            break

    if return_history:
        history["converged"] = converged
        history["n_iter"] = it + 1
        return F, Q, history
    else:
        return F, Q


class FitnessComplexity:
    """Scikit-learn-style estimator for Fitness-Complexity fixed-point iteration.

    This estimator computes country/user fitness and product/word complexity
    via the nonlinear Fitness-Complexity fixed-point iteration. It follows
    scikit-learn conventions: hyperparameters in __init__, data in fit().

    Parameters:
        n_iter: Maximum number of iterations (default: 200).
        tol: Convergence tolerance on max absolute change (default: 1e-10).
        verbose: If True, print convergence message (default: True).

    Attributes (after fit):
        fitness_: Fitted country/user fitness scores (n_rows,), normalized to mean 1.
        complexity_: Fitted product/word complexity scores (n_cols,), normalized to mean 1.
        n_iter_: Number of iterations performed.
        converged_: Whether algorithm converged.
        
    Notes:
        Following scikit-learn conventions, only minimal convergence diagnostics
        (n_iter_ and converged_) are stored as attributes. This matches the pattern
        used by sklearn.linear_model.LogisticRegression and similar iterative
        estimators.
        
        For detailed convergence history (per-iteration dF, dQ values), use the
        deprecated `fitness_complexity()` function with `return_history=True`. 
        This is intended for debugging and research, not typical usage.

    Examples:
        >>> from fitkit.algorithms import FitnessComplexity
        >>> fc = FitnessComplexity(n_iter=200, tol=1e-10)
        >>> fc.fit(M)  # M is binary incidence matrix
        >>> F = fc.fitness_
        >>> Q = fc.complexity_
        >>> print(f"Converged: {fc.converged_}, iterations: {fc.n_iter_}")

        >>> # Or: one-liner
        >>> F, Q = FitnessComplexity(n_iter=200).fit_transform(M)
        
        >>> # For detailed convergence diagnostics (debugging):
        >>> from fitkit.algorithms import fitness_complexity
        >>> F, Q, history = fitness_complexity(M, return_history=True)
        >>> # history contains: dF, dQ, converged, n_iter

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

    def fit(self, X: sp.spmatrix, y: np.ndarray | None = None):
        """Compute Fitness-Complexity fixed point on binary incidence matrix X.

        Args:
            X: Scipy sparse matrix (n_rows × n_cols), entries in {0,1}.
               Rows represent countries/users, columns represent products/words.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            self: Fitted estimator.
        """
        F, Q, history = _fitness_complexity(
            X, n_iter=self.n_iter, tol=self.tol, 
            return_history=True, verbose=self.verbose
        )
        
        self.fitness_ = F
        self.complexity_ = Q
        self.n_iter_ = history["n_iter"]
        self.converged_ = history["converged"]

        return self

    def fit_transform(self, X: sp.spmatrix, y: np.ndarray | None = None):
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


# Deprecated public function - use FitnessComplexity class instead
def fitness_complexity(
    M_bin: sp.spmatrix,
    n_iter: int = 200,
    tol: float = 1e-10,
    return_history: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Compute Fitness-Complexity fixed point (DEPRECATED).
    
    .. deprecated:: 0.2.0
        Use `FitnessComplexity` class instead (scikit-learn interface). This function
        will be removed in a future version.
        
        Note: This function remains useful for accessing detailed convergence
        history (per-iteration dF, dQ values) when `return_history=True`. The
        class interface only exposes minimal convergence info (n_iter_, converged_)
        following scikit-learn conventions.
        
    Args:
        M_bin: Binary incidence matrix (n_rows × n_cols)
        n_iter: Maximum number of iterations
        tol: Convergence tolerance
        return_history: Return convergence diagnostics (detailed history dict)
        
    Returns:
        fitness, complexity arrays (and history dict with per-iteration diagnostics)
        
    Example:
        >>> # Typical usage (class interface, recommended):
        >>> from fitkit.algorithms import FitnessComplexity
        >>> model = FitnessComplexity()
        >>> fitness, complexity = model.fit_transform(M)
        >>> print(f"Converged in {model.n_iter_} iterations")
        >>> 
        >>> # For detailed convergence diagnostics (debugging):
        >>> fitness, complexity, history = fitness_complexity(M, return_history=True)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(history['dF'], label='dF (fitness change)')
        >>> plt.plot(history['dQ'], label='dQ (complexity change)')
        >>> plt.legend()
    
    References:
        Lawrence, N.D. (2024). "Conditional Likelihood Interpretation of Economic Fitness".
        Tacchella et al. (2012). "A New Metrics for Countries' Fitness and Products' Complexity".
    """
    warnings.warn(
        "fitness_complexity() is deprecated. "
        "Use FitnessComplexity class instead (scikit-learn interface).",
        DeprecationWarning,
        stacklevel=2
    )
    return _fitness_complexity(
        M_bin, n_iter=n_iter, tol=tol, return_history=return_history, verbose=True
    )
