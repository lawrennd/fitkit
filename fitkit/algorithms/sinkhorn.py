"""Masked Sinkhorn-Knopp / Iterative Proportional Fitting (IPF) algorithm.

This module implements matrix scaling with support constraints, also known as
the Sinkhorn-Knopp algorithm or Iterative Proportional Fitting (IPF).

The relationship between Sinkhorn scaling and Fitness-Complexity is explored in:

    Lawrence, N.D. (2024). "Conditional Likelihood Interpretation of
    Economic Fitness" (working paper).

The algorithm finds diagonal scaling matrices U and V such that:
    W = U @ K @ V
matches prescribed row and column marginals, where K is a binary support mask.
"""

import numpy as np
import scipy.sparse as sp


def sinkhorn_masked(
    M_bin: sp.spmatrix,
    r: np.ndarray,
    c: np.ndarray,
    n_iter: int = 2000,
    tol: float = 1e-12,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray, sp.spmatrix, dict]:
    """Sinkhorn/IPF scaling on a support mask.

    Given a binary support matrix K (typically M_bin, the incidence matrix),
    find diagonal scalings u, v such that:

        W = diag(u) @ K @ diag(v)

    matches the desired row marginals r and column marginals c:
        W.sum(axis=1) ≈ r
        W.sum(axis=0) ≈ c

    This is the Sinkhorn-Knopp algorithm / Iterative Proportional Fitting (IPF),
    constrained to the support of K (entries off-support remain zero).

    Args:
        M_bin: Scipy sparse matrix (n_rows × n_cols), entries in {0,1}.
               This defines the support (where entries can be nonzero).
        r: Desired row marginals (n_rows,), must have positive total mass.
        c: Desired column marginals (n_cols,), must have positive total mass.
        n_iter: Maximum number of iterations (default: 2000).
        tol: Convergence tolerance on max marginal error (default: 1e-12).
        eps: Numerical guard to avoid division by zero (default: 1e-30).

    Returns:
        u: Row scaling factors (n_rows,).
        v: Column scaling factors (n_cols,).
        W: Scaled sparse matrix W = diag(u) @ K @ diag(v) (sparse).
        history: Dict with convergence diagnostics:
            - "dr": list of max row marginal errors per iteration
            - "dc": list of max column marginal errors per iteration
            - "iters": number of iterations performed
            - "converged": True if converged within tolerance

    Raises:
        ValueError: If marginals are infeasible (e.g., positive mass on isolated nodes).

    Notes:
        - This implementation is **sparse-safe**: it never densifies K or W.
        - Row and column marginals are automatically normalized to have equal total mass.
        - Convergence is typically fast (tens to hundreds of iterations) for feasible problems.
        - If the support graph is disconnected, the solution may not be unique.

    References:
        Sinkhorn & Knopp (1967). "Concerning nonnegative matrices and doubly stochastic matrices".
        Lawrence, N.D. (2024). "Conditional Likelihood Interpretation of Economic Fitness".
    """
    K = M_bin.tocsr()  # Ensure CSR format for efficient row operations

    r = np.asarray(r, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()

    if K.shape[0] != r.shape[0] or K.shape[1] != c.shape[0]:
        raise ValueError(f"Shape mismatch: K is {K.shape}, r is {r.shape}, c is {c.shape}.")

    # Normalize marginals to have equal total mass
    rs = float(r.sum())
    cs = float(c.sum())
    if rs <= 0 or cs <= 0:
        raise ValueError("r and c must have positive total mass")
    if abs(rs - cs) > 1e-12 * max(rs, cs):
        c = c * (rs / cs)

    # Feasibility check: isolated nodes with positive mass are infeasible
    row_deg = np.asarray(K.sum(axis=1)).ravel()
    col_deg = np.asarray(K.sum(axis=0)).ravel()
    if np.any((row_deg == 0) & (r > 0)):
        raise ValueError("Infeasible (r>0 on a row with zero support)")
    if np.any((col_deg == 0) & (c > 0)):
        raise ValueError("Infeasible (c>0 on a col with zero support)")

    # Initialize scaling factors
    u = np.ones(K.shape[0], dtype=float)
    v = np.ones(K.shape[1], dtype=float)

    history = {"dr": [], "dc": [], "iters": 0, "converged": False}

    for it in range(n_iter):
        # Update u: scale rows to match r
        Kv = K @ v
        u_new = r / np.maximum(Kv, eps)

        # Update v: scale columns to match c
        KTu = K.T @ u_new
        v_new = c / np.maximum(KTu, eps)

        # Compute marginal errors without forming W explicitly
        r_hat = u_new * (K @ v_new)
        c_hat = v_new * (K.T @ u_new)

        dr = float(np.max(np.abs(r_hat - r)))
        dc = float(np.max(np.abs(c_hat - c)))
        history["dr"].append(dr)
        history["dc"].append(dc)
        history["iters"] = it + 1

        u, v = u_new, v_new

        # Check convergence
        if max(dr, dc) < tol:
            history["converged"] = True
            break

    # Construct scaled matrix W = diag(u) @ K @ diag(v) (sparse)
    W = sp.diags(u) @ K @ sp.diags(v)
    return u, v, W, history
