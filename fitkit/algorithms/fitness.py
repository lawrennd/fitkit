"""Fitness-Complexity fixed-point iteration.

This module implements the nonlinear Fitness-Complexity algorithm
as described in:

    Lawrence, N.D. (2024). "Conditional Likelihood Interpretation of
    Economic Fitness" (working paper).

The algorithm computes country fitness F and product complexity Q
via alternating harmonic aggregation on the bipartite support graph.
"""

import numpy as np
import scipy.sparse as sp


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
