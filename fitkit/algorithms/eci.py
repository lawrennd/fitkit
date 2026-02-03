"""Economic Complexity Index (ECI) and Product Complexity Index (PCI).

This module implements the spectral/linear ECI/PCI algorithm, which serves
as a baseline for comparison with the nonlinear Fitness-Complexity method.

The ECI/PCI are computed via the eigenvector centrality of the country-country
projection matrix, as described in:

    Hidalgo, C.A. & Hausmann, R. (2009). "The building blocks of economic
    complexity". PNAS 106(26): 10570-10575.

This implementation follows the standard formulation and is provided as a
first-class baseline per the tenet "baselines-are-first-class".
"""

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
        pci: Product PCI scores (n_products,), standardized to mean 0, std 1.

    Raises:
        ValueError: If the matrix has insufficient structure for ECI computation
                   (e.g., isolated countries/products).

    Notes:
        - The sign of ECI is arbitrary; it is fixed by correlating positively
          with diversification (k_c).
        - Isolated nodes (zero degree) are dropped before computation.
        - This is a **linear** method (spectral) in contrast to the nonlinear
          Fitness-Complexity fixed point.

    References:
        Hidalgo & Hausmann (2009). "The building blocks of economic complexity". PNAS.
        Mariani et al. (2015). "Measuring Economic Complexity of Countries and Products".
    """
    # Convert to dense for eigen-decomposition (can be optimized with sparse eigs if needed)
    Mv = M_bin.toarray()
    kc = Mv.sum(axis=1)  # Country diversification
    kp = Mv.sum(axis=0)  # Product ubiquity

    # Drop zero-degree nodes (isolated countries/products)
    keep_c = kc > 0
    keep_p = kp > 0
    Mv = Mv[keep_c][:, keep_p]
    kc = kc[keep_c]
    kp = kp[keep_p]

    # Normalized projection matrices
    Dc_inv = np.diag(1.0 / kc)
    Dp_inv = np.diag(1.0 / kp)

    # Country-country projection: C = (M/kc) @ (M^T/kp)
    C = Dc_inv @ Mv @ Dp_inv @ Mv.T

    # Eigen-decomposition (C is symmetric)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]  # Sort by descending eigenvalue
    evals = evals[order]
    evecs = evecs[:, order]

    if evecs.shape[1] < 2:
        raise ValueError("Not enough dimensions for ECI (need at least 2 eigenvectors).")

    # ECI is the second eigenvector (first is trivial: uniform)
    eci_vec = evecs[:, 1]

    # Fix sign: ECI should correlate positively with diversification
    if np.corrcoef(eci_vec, kc)[0, 1] < 0:
        eci_vec = -eci_vec

    # PCI: projection back to product space
    pci_vec = Dp_inv @ Mv.T @ eci_vec

    # Standardize for convenience (mean 0, std 1)
    eci = (eci_vec - eci_vec.mean()) / (eci_vec.std(ddof=0) + 1e-12)
    pci = (pci_vec - pci_vec.mean()) / (pci_vec.std(ddof=0) + 1e-12)

    return eci, pci
