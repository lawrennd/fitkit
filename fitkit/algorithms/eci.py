"""Economic Complexity Index (ECI) and Product Complexity Index (PCI).

This module implements the spectral/linear ECI/PCI algorithm, which serves
as a baseline for comparison with the nonlinear Fitness-Complexity method.

The ECI/PCI are computed via the eigenvector centrality of the country-country
projection matrix, as described in:

    Hidalgo, C.A. & Hausmann, R. (2009). "The building blocks of economic
    complexity". PNAS 106(26): 10570-10575.

This implementation follows the standard formulation and is provided as a
first-class baseline per the tenet "baselines-are-first-class".

Provides both a scikit-learn-style estimator (ECI) and a functional API
(compute_eci_pci) for convenience and backward compatibility.
"""

import warnings
import numpy as np
import scipy.sparse as sp


def compute_eci_pci(M_bin: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Compute ECI/PCI from binary matrix using the standard spectral formulation.

    The algorithm constructs the country-country projection matrix:
        C = (M / k_c) @ (M^T / k_p)
    where k_c and k_p are the row and column degrees (diversification and ubiquity).

    ECI is the second eigenvector of C (the first is trivial), and PCI is
    the projection of ECI back to the product space.

    Args:
        M_bin: Scipy sparse matrix (n_countries × n_products), entries in {0,1}.

    Returns:
        eci: Country ECI scores (n_countries,), standardized to mean 0, std 1.
             Isolated countries (zero diversification) will have NaN values.
        pci: Product PCI scores (n_products,), standardized to mean 0, std 1.
             Isolated products (zero ubiquity) will have NaN values.

    Raises:
        ValueError: If the matrix has insufficient structure for ECI computation
                   (e.g., too few connected nodes for eigenvalue decomposition).

    Warnings:
        UserWarning: If isolated nodes (zero diversification/ubiquity) are found,
                    a warning is issued indicating how many nodes have NaN values.

    Notes:
        - The sign of ECI is arbitrary; it is fixed by correlating positively
          with diversification (k_c).
        - Isolated nodes (zero degree) receive NaN values in the output.
        - This is a **linear** method (spectral) in contrast to the nonlinear
          Fitness-Complexity fixed point.

    References:
        Hidalgo & Hausmann (2009). "The building blocks of economic complexity". PNAS.
        Mariani et al. (2015). "Measuring Economic Complexity of Countries and Products".
    """
    # Convert to dense for eigen-decomposition (can be optimized with sparse eigs if needed)
    Mv = M_bin.toarray()
    n_countries_orig = Mv.shape[0]
    n_products_orig = Mv.shape[1]
    
    kc = Mv.sum(axis=1)  # Country diversification
    kp = Mv.sum(axis=0)  # Product ubiquity

    # Identify zero-degree nodes (isolated countries/products)
    keep_c = kc > 0
    keep_p = kp > 0
    
    n_dropped_c = (~keep_c).sum()
    n_dropped_p = (~keep_p).sum()
    
    # Warn user if nodes are dropped
    if n_dropped_c > 0 or n_dropped_p > 0:
        warnings.warn(
            f"ECI: Dropped {n_dropped_c} isolated countries (zero diversification) "
            f"and {n_dropped_p} isolated products (zero ubiquity). "
            f"These will have NaN values in output.",
            UserWarning,
            stacklevel=2
        )
    
    # Extract connected component
    Mv_conn = Mv[keep_c][:, keep_p]
    kc_conn = kc[keep_c]
    kp_conn = kp[keep_p]

    # Normalized projection matrices
    Dc_inv = np.diag(1.0 / kc_conn)
    Dp_inv = np.diag(1.0 / kp_conn)

    # Country-country projection: C = (M/kc) @ (M^T/kp)
    C = Dc_inv @ Mv_conn @ Dp_inv @ Mv_conn.T

    # Eigen-decomposition (C is symmetric)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]  # Sort by descending eigenvalue
    evals = evals[order]
    evecs = evecs[:, order]

    if evecs.shape[1] < 2:
        raise ValueError("Not enough dimensions for ECI (need at least 2 eigenvectors).")

    # ECI is the second eigenvector (first is trivial: uniform)
    eci_vec_conn = evecs[:, 1]

    # Fix sign: ECI should correlate positively with diversification
    if np.corrcoef(eci_vec_conn, kc_conn)[0, 1] < 0:
        eci_vec_conn = -eci_vec_conn

    # PCI: projection back to product space
    pci_vec_conn = Dp_inv @ Mv_conn.T @ eci_vec_conn

    # Standardize for convenience (mean 0, std 1)
    eci_conn = (eci_vec_conn - eci_vec_conn.mean()) / (eci_vec_conn.std(ddof=0) + 1e-12)
    pci_conn = (pci_vec_conn - pci_vec_conn.mean()) / (pci_vec_conn.std(ddof=0) + 1e-12)
    
    # Expand back to original dimensions with NaN for dropped nodes
    eci = np.full(n_countries_orig, np.nan)
    pci = np.full(n_products_orig, np.nan)
    eci[keep_c] = eci_conn
    pci[keep_p] = pci_conn

    return eci, pci


class ECI:
    """Scikit-learn-style estimator for Economic Complexity Index (ECI) and
    Product Complexity Index (PCI).

    This estimator computes ECI/PCI via the spectral/linear method (second eigenvector
    of the country-country projection matrix). It serves as a baseline for comparison
    with the nonlinear Fitness-Complexity method.

    Parameters:
        None (ECI/PCI computation has no hyperparameters).

    Attributes (set after calling fit):
        eci_: Country ECI scores (n_countries,), standardized to mean 0, std 1.
        pci_: Product PCI scores (n_products,), standardized to mean 0, std 1.

    Examples:
        >>> from fitkit.algorithms import ECI
        >>> eci_estimator = ECI()
        >>> eci_estimator.fit(M)  # M is binary incidence matrix
        >>> eci = eci_estimator.eci_
        >>> pci = eci_estimator.pci_

        >>> # Or: one-liner
        >>> eci, pci = ECI().fit_transform(M)

    References:
        Hidalgo & Hausmann (2009). "The building blocks of economic complexity". PNAS.
        Mariani et al. (2015). "Measuring Economic Complexity of Countries and Products".
    """

    def __init__(self):
        """Initialize ECI estimator.

        Note: ECI/PCI computation has no hyperparameters.
        """
        pass

    def fit(self, X: sp.spmatrix, y: np.ndarray | None = None):
        """Compute ECI/PCI on binary incidence matrix X.

        Args:
            X: Scipy sparse matrix (n_countries × n_products), entries in {0,1}.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            self: Fitted estimator.
        """
        self.eci_, self.pci_ = compute_eci_pci(X)
        return self

    def fit_transform(self, X: sp.spmatrix, y: np.ndarray | None = None):
        """Fit and return (ECI, PCI).

        Args:
            X: Scipy sparse matrix (n_countries × n_products), entries in {0,1}.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            eci: Country ECI scores (n_countries,).
            pci: Product PCI scores (n_products,).
        """
        self.fit(X, y)
        return self.eci_, self.pci_
