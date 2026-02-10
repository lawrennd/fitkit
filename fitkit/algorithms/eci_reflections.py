"""Method of Reflections for Economic Complexity Index (ECI) and Product Complexity Index (PCI).

This module implements the iterative Method of Reflections algorithm from
Hidalgo & Hausmann (2009), which should theoretically converge to the same
eigenvector as the direct eigenvalue decomposition in `eci.py`.

**WARNING**: This method can fail or converge slowly when the eigenvalue gap
(λ₀ - λ₁) is small. Use `ECIReflections` class (scikit-learn interface) which
includes built-in diagnostics.

References:
    Hidalgo, C.A. & Hausmann, R. (2009). "The building blocks of economic
    complexity". PNAS 106(26): 10570-10575.
"""

import warnings
import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple, Dict


def _check_eigengap(M_bin: sp.spmatrix, verbose: bool = True) -> Dict[str, float]:
    """Diagnose eigenvalue gap to predict Method of Reflections convergence (private).
    
    Small eigengap → slow convergence or failure!
    Zero eigengap (degenerate) → power iteration undefined!
    
    Args:
        M_bin: Binary incidence matrix (n_countries × n_products)
        verbose: If True, print diagnostic information
        
    Returns:
        dict with eigenvalues, eigengap, and convergence estimate
        
    Note:
        This is a private function. Use ECIReflections.check_eigengap() instead.
    """
    Mv = M_bin.toarray()
    kc = Mv.sum(axis=1)
    kp = Mv.sum(axis=0)
    
    # Filter zero-degree nodes
    keep_c = kc > 0
    keep_p = kp > 0
    Mv_conn = Mv[keep_c][:, keep_p]
    kc_conn = kc[keep_c]
    kp_conn = kp[keep_p]
    
    # Construct projection matrix
    Dc_inv = np.diag(1.0 / kc_conn)
    Dp_inv = np.diag(1.0 / kp_conn)
    C = Dc_inv @ Mv_conn @ Dp_inv @ Mv_conn.T
    
    # Compute eigenvalues
    evals = np.linalg.eigvalsh(C)
    evals = np.sort(evals)[::-1]
    
    eigengap = evals[0] - evals[1]
    ratio = evals[1] / evals[0] if evals[0] > 1e-10 else np.nan
    
    # Estimate convergence rate (simple formula: iterations ~ log(ε) / log(ratio))
    if ratio < 0.999 and not np.isnan(ratio):
        iters_99pct = int(np.log(0.01) / np.log(ratio))
        iters_999pct = int(np.log(0.001) / np.log(ratio))
    else:
        iters_99pct = np.inf
        iters_999pct = np.inf
    
    result = {
        'eigenvalue_0': evals[0],
        'eigenvalue_1': evals[1],
        'eigengap': eigengap,
        'ratio_lambda1_lambda0': ratio,
        'iterations_for_99pct': iters_99pct,
        'iterations_for_999pct': iters_999pct,
        'is_degenerate': eigengap < 1e-6,
    }
    
    if verbose:
        print("=" * 70)
        print("EIGENGAP DIAGNOSTIC FOR METHOD OF REFLECTIONS")
        print("=" * 70)
        print(f"Top eigenvalues: λ₀ = {evals[0]:.6f}, λ₁ = {evals[1]:.6f}")
        print(f"Eigengap: {eigengap:.6f}")
        print(f"Ratio λ₁/λ₀: {ratio:.6f}")
        
        if result['is_degenerate']:
            print("\n⚠️  DEGENERATE EIGENSPACE!")
            print("    Eigengap near zero → power iteration undefined")
            print("    Method of Reflections will FAIL or give random results")
            print("    → Use direct eigenvalue method instead!")
        elif iters_99pct < 20:
            print(f"\n✓ Good eigengap - should converge in ~{iters_99pct} iterations")
        elif iters_99pct < 100:
            print(f"\n⚠️  Moderate eigengap - may need {iters_99pct}+ iterations")
        else:
            print(f"\n⚠️  Small eigengap - convergence will be very slow (>{iters_99pct} iters)")
        
        print("=" * 70)
    
    return result


def _compute_eci_pci_reflections(
    M_bin: sp.spmatrix,
    max_iter: int = 200,
    tolerance: float = 1e-6,
    check_eigengap_first: bool = True,
    return_history: bool = False
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict]:
    """Compute ECI/PCI using iterative Method of Reflections (private function).
    
    **WARNING**: This method can fail when eigenvalue gap is small!
    Set `check_eigengap_first=True` to diagnose potential issues.
    
    Note:
        This is a private function. Use ECIReflections class instead.
    
    The algorithm iteratively updates country and product complexity:
        k_c[n] = (1/k_c[0]) * M @ k_p[n-1]
        k_p[n] = (1/k_p[0]) * M^T @ k_c[n]
    
    Starting from k_c[0] = diversification, k_p[0] = ubiquity.
    
    **Theory**: Should converge to second eigenvector of projection matrix
    (same as eigenvalue-based method in `eci.py`).
    
    **Practice**: Can fail or converge to wrong solution when:
    - Eigenvalue gap is small (slow convergence)
    - Eigenvalues are degenerate (undefined behavior)
    - Numerical precision issues accumulate
    
    Args:
        M_bin: Binary incidence matrix (n_countries × n_products)
        max_iter: Maximum iterations (default 200)
        tolerance: Convergence threshold for relative change
        check_eigengap_first: If True, check eigengap and warn if problematic
        return_history: If True, return convergence history for diagnostics
        
    Returns:
        eci: Country complexity scores (standardized)
        pci: Product complexity scores (standardized)
        history (optional): Dict with convergence diagnostics
        
    Raises:
        ValueError: If eigengap is degenerate and method will fail
        RuntimeWarning: If convergence is slow or unsuccessful
    """
    # Check eigengap first to predict convergence issues
    if check_eigengap_first:
        gap_info = _check_eigengap(M_bin, verbose=False)
        if gap_info['is_degenerate']:
            raise ValueError(
                f"Eigenvalue gap too small ({gap_info['eigengap']:.2e}). "
                "Method of Reflections is undefined for degenerate eigenspaces. "
                "Use compute_eci_pci() (direct eigenvalue method) instead."
            )
        if gap_info['iterations_for_99pct'] > max_iter:
            warnings.warn(
                f"Small eigengap detected (λ₀-λ₁ = {gap_info['eigengap']:.4f}). "
                f"Estimated {int(gap_info['iterations_for_99pct'])} iterations "
                f"for convergence, but max_iter={max_iter}. Consider increasing.",
                RuntimeWarning,
                stacklevel=2
            )
    
    # Convert to dense and filter zero-degree nodes
    Mv = M_bin.toarray()
    n_countries_orig = Mv.shape[0]
    n_products_orig = Mv.shape[1]
    
    kc = Mv.sum(axis=1)
    kp = Mv.sum(axis=0)
    
    keep_c = kc > 0
    keep_p = kp > 0
    
    n_dropped_c = (~keep_c).sum()
    n_dropped_p = (~keep_p).sum()
    
    if n_dropped_c > 0 or n_dropped_p > 0:
        warnings.warn(
            f"Reflections: Dropped {n_dropped_c} isolated countries and "
            f"{n_dropped_p} isolated products (zero degree).",
            UserWarning,
            stacklevel=2
        )
    
    Mv_conn = Mv[keep_c][:, keep_p]
    kc_conn = kc[keep_c]
    kp_conn = kp[keep_p]
    
    # Initialize with diversification/ubiquity
    k_c_prev = kc_conn.copy()
    k_p_prev = kp_conn.copy()
    
    # Normalization factors
    norm_c = kc_conn
    norm_p = kp_conn
    
    # History tracking
    if return_history:
        history = {
            'iterations': [],
            'delta_c': [],
            'delta_p': [],
            'k_c_norm': [],
            'k_p_norm': []
        }
    
    # Iterative updates
    converged = False
    for iteration in range(1, max_iter + 1):
        # Update country complexity from product complexity
        k_c_new = (Mv_conn @ k_p_prev) / norm_c
        
        # Update product complexity from country complexity
        k_p_new = (Mv_conn.T @ k_c_new) / norm_p
        
        # Check convergence
        delta_c = np.linalg.norm(k_c_new - k_c_prev) / (np.linalg.norm(k_c_new) + 1e-12)
        delta_p = np.linalg.norm(k_p_new - k_p_prev) / (np.linalg.norm(k_p_new) + 1e-12)
        
        if return_history:
            history['iterations'].append(iteration)
            history['delta_c'].append(delta_c)
            history['delta_p'].append(delta_p)
            history['k_c_norm'].append(np.linalg.norm(k_c_new))
            history['k_p_norm'].append(np.linalg.norm(k_p_new))
        
        # Check if both converged
        if delta_c < tolerance and delta_p < tolerance:
            converged = True
            break
        
        k_c_prev = k_c_new
        k_p_prev = k_p_new
    
    if not converged:
        warnings.warn(
            f"Method of Reflections did not converge after {max_iter} iterations. "
            f"Final deltas: Δc={delta_c:.2e}, Δp={delta_p:.2e}. "
            f"Results may be unreliable. Check eigengap with check_eigengap().",
            RuntimeWarning,
            stacklevel=2
        )
    
    # Final values
    eci_vec_conn = k_c_new
    pci_vec_conn = k_p_new
    
    # Fix sign: correlate with diversification (same convention as eigenvalue method)
    if np.corrcoef(eci_vec_conn, kc_conn)[0, 1] < 0:
        eci_vec_conn = -eci_vec_conn
    
    # Standardize (mean 0, std 1)
    eci_conn = (eci_vec_conn - eci_vec_conn.mean()) / (eci_vec_conn.std(ddof=0) + 1e-12)
    pci_conn = (pci_vec_conn - pci_vec_conn.mean()) / (pci_vec_conn.std(ddof=0) + 1e-12)
    
    # Expand back to original dimensions with NaN for dropped nodes
    eci = np.full(n_countries_orig, np.nan)
    pci = np.full(n_products_orig, np.nan)
    eci[keep_c] = eci_conn
    pci[keep_p] = pci_conn
    
    if return_history:
        history['converged'] = converged
        history['final_iteration'] = iteration
        return eci, pci, history
    else:
        return eci, pci


class ECIReflections:
    """Scikit-learn-style estimator for ECI/PCI using Method of Reflections.
    
    **WARNING**: This iterative method can fail when eigenvalue gap is small!
    Use `check_eigengap()` first to diagnose potential convergence issues.
    
    For most applications, prefer the direct eigenvalue method in `eci.py`.
    
    Parameters:
        max_iter: Maximum iterations (default 200)
        tolerance: Convergence threshold (default 1e-6)
        check_eigengap_first: Check eigengap before running (default True)
        
    Attributes (after fit):
        eci_: Country ECI scores
        pci_: Product PCI scores
        n_iter_: Number of iterations used
        converged_: Whether algorithm converged
        
    Notes:
        Following scikit-learn conventions, only minimal convergence diagnostics
        (n_iter_ and converged_) are stored as attributes. This matches the pattern
        used by sklearn.linear_model.LogisticRegression and similar iterative
        estimators.
        
        For detailed convergence history (per-iteration deltas, norms, etc.),
        use the deprecated `compute_eci_pci_reflections()` function with
        `return_history=True`. This is intended for debugging and research, not
        typical usage.
        
    Examples:
        >>> from fitkit.algorithms import ECIReflections
        >>> estimator = ECIReflections(max_iter=200)
        >>> estimator.fit(M)  # M is binary incidence matrix
        >>> eci = estimator.eci_
        >>> pci = estimator.pci_
        >>> print(f"Converged: {estimator.converged_}, iterations: {estimator.n_iter_}")
        
        >>> # Or: one-liner
        >>> eci, pci = ECIReflections(max_iter=200).fit_transform(M)
        
        >>> # For detailed convergence diagnostics (debugging):
        >>> from fitkit.algorithms import compute_eci_pci_reflections
        >>> eci, pci, history = compute_eci_pci_reflections(M, return_history=True)
        >>> # history contains: iterations, delta_c, delta_p, k_c_norm, k_p_norm
    """
    
    def __init__(
        self,
        max_iter: int = 200,
        tolerance: float = 1e-6,
        check_eigengap_first: bool = True
    ):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.check_eigengap_first = check_eigengap_first
        
        # Attributes set by fit()
        self.eci_: Optional[np.ndarray] = None
        self.pci_: Optional[np.ndarray] = None
        self.n_iter_: Optional[int] = None
        self.converged_: Optional[bool] = None
    
    def fit(self, X: sp.spmatrix, y: np.ndarray | None = None):
        """Fit the Method of Reflections to compute ECI/PCI.
        
        Args:
            X: Binary incidence matrix (n_countries × n_products)
            y: Ignored. Present for sklearn compatibility.
            
        Returns:
            self: Fitted estimator.
        """
        eci, pci, history = _compute_eci_pci_reflections(
            X,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            check_eigengap_first=self.check_eigengap_first,
            return_history=True
        )
        
        self.eci_ = eci
        self.pci_ = pci
        self.n_iter_ = history['final_iteration']
        self.converged_ = history['converged']
        
        return self
    
    def fit_transform(self, X: sp.spmatrix, y: np.ndarray | None = None):
        """Fit estimator and return ECI/PCI arrays.
        
        Args:
            X: Binary incidence matrix (n_countries × n_products)
            y: Ignored. Present for sklearn compatibility.
            
        Returns:
            eci: Country ECI scores (n_countries,)
            pci: Product PCI scores (n_products,)
        """
        self.fit(X, y)
        return self.eci_, self.pci_
    
    @staticmethod
    def check_eigengap(M_bin: sp.spmatrix, verbose: bool = True) -> Dict[str, float]:
        """Diagnose eigenvalue gap to predict convergence behavior.
        
        Small eigengap → slow convergence or failure!
        Zero eigengap (degenerate) → power iteration undefined!
        
        Args:
            M_bin: Binary incidence matrix (n_countries × n_products)
            verbose: If True, print diagnostic information
            
        Returns:
            dict with eigenvalues, eigengap, and convergence estimate
            
        Example:
            >>> from fitkit.algorithms import ECIReflections
            >>> gap_info = ECIReflections.check_eigengap(M_bin)
        """
        return _check_eigengap(M_bin, verbose=verbose)
    
    def __repr__(self):
        return (
            f"ECIReflections(max_iter={self.max_iter}, "
            f"tolerance={self.tolerance}, "
            f"check_eigengap_first={self.check_eigengap_first})"
        )


# Deprecated public functions - use ECIReflections class instead
def check_eigengap(M_bin: sp.spmatrix, verbose: bool = True) -> Dict[str, float]:
    """Diagnose eigenvalue gap (DEPRECATED).
    
    .. deprecated:: 0.2.0
        Use `ECIReflections.check_eigengap()` instead. This function will be
        removed in a future version.
    
    Args:
        M_bin: Binary incidence matrix
        verbose: If True, print diagnostic information
        
    Returns:
        dict with eigenvalue diagnostics
    """
    warnings.warn(
        "check_eigengap() is deprecated. Use ECIReflections.check_eigengap() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _check_eigengap(M_bin, verbose=verbose)


def compute_eci_pci_reflections(
    M_bin: sp.spmatrix,
    max_iter: int = 200,
    tolerance: float = 1e-6,
    check_eigengap_first: bool = True,
    return_history: bool = False
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict]:
    """Compute ECI/PCI using Method of Reflections (DEPRECATED).
    
    .. deprecated:: 0.2.0
        Use `ECIReflections` class instead (scikit-learn interface). This function
        will be removed in a future version.
        
        Note: This function remains useful for accessing detailed convergence
        history (per-iteration deltas, norms) when `return_history=True`. The
        class interface only exposes minimal convergence info (n_iter_, converged_)
        following scikit-learn conventions.
        
    Args:
        M_bin: Binary incidence matrix
        max_iter: Maximum iterations
        tolerance: Convergence threshold
        check_eigengap_first: Check eigengap before iterating
        return_history: Return convergence diagnostics (detailed history dict)
        
    Returns:
        eci, pci arrays (and optionally history dict with per-iteration diagnostics)
        
    Example:
        >>> # Typical usage (class interface, recommended):
        >>> from fitkit.algorithms import ECIReflections
        >>> model = ECIReflections()
        >>> eci, pci = model.fit_transform(M_bin)
        >>> print(f"Converged in {model.n_iter_} iterations")
        >>> 
        >>> # For detailed convergence diagnostics (debugging):
        >>> eci, pci, history = compute_eci_pci_reflections(M_bin, return_history=True)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(history['iterations'], history['delta_c'], label='delta_c')
        >>> plt.plot(history['iterations'], history['delta_p'], label='delta_p')
    """
    warnings.warn(
        "compute_eci_pci_reflections() is deprecated. "
        "Use ECIReflections class instead (scikit-learn interface).",
        DeprecationWarning,
        stacklevel=2
    )
    return _compute_eci_pci_reflections(
        M_bin, max_iter=max_iter, tolerance=tolerance,
        check_eigengap_first=check_eigengap_first, return_history=return_history
    )
