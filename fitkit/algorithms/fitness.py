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
        min_ubiquity: Minimum number of countries/users that must export/use a product/word
            for it to be included (default: 3). Products with lower ubiquity are more likely to 
            create isolated subgraphs that violate the connectivity assumption. 
        min_diversification: Minimum number of products/words that a country/user must 
            export/use to be included (default: 5). Countries with lower diversification
            may create disconnected components.
        iterative_filter: If True, iteratively reapply filters until matrix size stabilizes
            (default: True). Removing nodes can cascade: a filtered product may leave a country
            below min_diversification, which then needs filtering, which may affect other
            products, etc.

    Attributes (after fit):
        fitness_: Fitted country/user fitness scores (n_rows,), normalized to mean 1.
            For filtered entities, fitness is NaN.
        complexity_: Fitted product/word complexity scores (n_cols,), normalized to mean 1.
            For filtered entities, complexity is NaN.
        n_iter_: Number of iterations performed.
        converged_: Whether algorithm converged.
        country_mask_: Boolean array indicating which countries/users were kept (not filtered).
        product_mask_: Boolean array indicating which products/words were kept (not filtered).
        n_countries_filtered_: Number of countries/users filtered out.
        n_products_filtered_: Number of products/words filtered out.
        
    Notes:
        Following scikit-learn conventions, only minimal convergence diagnostics
        (n_iter_ and converged_) are stored as attributes. This matches the pattern
        used by sklearn.linear_model.LogisticRegression and similar iterative
        estimators.
        
        *Filtering for numerical stability*: The Fitness-Complexity algorithm 
        mathematically requires the bipartite graph to be **connected**. Disconnected
        components or isolated subgraphs violate this assumption, causing numerical
        collapse where one component dominates (e.g., one country gets all the fitness).
        
        By default, the estimator filters out products with ubiquity < 3 and countries
        with diversification < 5. This heuristic prevents the most common cause of 
        disconnection: products exported by only 1-2 countries create isolated 2-node
        subgraphs. Filtering is standard practice in the economic complexity literature.
        
        Set `min_ubiquity=1` and `min_diversification=1` to disable filtering (not 
        recommended for real-world data, as it can cause severe numerical instability).
        
        For detailed convergence history (per-iteration dF, dQ values), use the
        deprecated `fitness_complexity()` function with `return_history=True`. 
        This is intended for debugging and research, not typical usage.

    Examples:
        >>> from fitkit.algorithms import FitnessComplexity
        >>> # Basic usage with default filtering (recommended)
        >>> fc = FitnessComplexity(n_iter=200, tol=1e-10)
        >>> fc.fit(M)  # M is binary incidence matrix
        >>> F = fc.fitness_
        >>> Q = fc.complexity_
        >>> print(f"Converged: {fc.converged_}, iterations: {fc.n_iter_}")
        >>> print(f"Filtered {fc.n_countries_filtered_} countries, {fc.n_products_filtered_} products")

        >>> # Or: one-liner
        >>> F, Q = FitnessComplexity(n_iter=200).fit_transform(M)
        
        >>> # Conservative filtering for high-density networks
        >>> fc = FitnessComplexity(min_ubiquity=5, min_diversification=10)
        >>> F, Q = fc.fit_transform(M)
        
        >>> # Disable filtering (not recommended, may cause numerical collapse)
        >>> fc = FitnessComplexity(min_ubiquity=1, min_diversification=1)
        >>> F, Q = fc.fit_transform(M)
        
        >>> # For detailed convergence diagnostics (debugging):
        >>> from fitkit.algorithms import fitness_complexity
        >>> F, Q, history = fitness_complexity(M, return_history=True)
        >>> # history contains: dF, dQ, converged, n_iter

    References:
        Lawrence, N.D. (2024). "Conditional Likelihood Interpretation of Economic Fitness".
        Tacchella et al. (2012). "A New Metrics for Countries' Fitness and Products' Complexity".
    """

    def __init__(
        self, 
        n_iter: int = 200, 
        tol: float = 1e-10, 
        verbose: bool = True,
        min_ubiquity: int = 3,
        min_diversification: int = 5,
        iterative_filter: bool = True
    ):
        """Initialize FitnessComplexity estimator.

        Args:
            n_iter: Maximum number of iterations.
            tol: Convergence tolerance.
            verbose: If True, print convergence message.
            min_ubiquity: Minimum ubiquity (countries per product) to keep product (default: 3).
                Products exported by fewer countries are filtered to prevent numerical collapse.
            min_diversification: Minimum diversification (products per country) to keep country (default: 5).
                Countries exporting fewer products are filtered.
            iterative_filter: If True, iteratively reapply filters until stable (default: True).
        """
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.min_ubiquity = min_ubiquity
        self.min_diversification = min_diversification
        self.iterative_filter = iterative_filter

    def fit(self, X: sp.spmatrix, y: np.ndarray | None = None):
        """Compute Fitness-Complexity fixed point on binary incidence matrix X.

        Args:
            X: Scipy sparse matrix (n_rows × n_cols), entries in {0,1}.
               Rows represent countries/users, columns represent products/words.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            self: Fitted estimator.
        """
        n_countries_orig, n_products_orig = X.shape
        
        # Apply filtering to remove low-connectivity nodes
        M_filtered, country_mask, product_mask = self._filter_matrix(X)
        
        # Store masks for reference
        self.country_mask_ = country_mask
        self.product_mask_ = product_mask
        self.n_countries_filtered_ = int((~country_mask).sum())
        self.n_products_filtered_ = int((~product_mask).sum())
        
        if self.verbose and (self.n_countries_filtered_ > 0 or self.n_products_filtered_ > 0):
            print(f"Filtered {self.n_countries_filtered_} countries, "
                  f"{self.n_products_filtered_} products due to low connectivity")
        
        # Check if filtered matrix is empty or too small
        n_countries_filt, n_products_filt = M_filtered.shape
        if n_countries_filt == 0 or n_products_filt == 0:
            if self.verbose:
                print("Warning: Filtering removed all nodes. Returning NaN arrays.")
            self.fitness_ = np.full(n_countries_orig, np.nan)
            self.complexity_ = np.full(n_products_orig, np.nan)
            self.n_iter_ = 0
            self.converged_ = False
            return self
        
        # Run algorithm on filtered matrix
        F_filtered, Q_filtered, history = _fitness_complexity(
            M_filtered, n_iter=self.n_iter, tol=self.tol, 
            return_history=True, verbose=self.verbose
        )
        
        # Expand results back to original dimensions (NaN for filtered nodes)
        self.fitness_ = self._expand_to_original(F_filtered, country_mask, n_countries_orig)
        self.complexity_ = self._expand_to_original(Q_filtered, product_mask, n_products_orig)
        self.n_iter_ = history["n_iter"]
        self.converged_ = history["converged"]

        return self

    def _filter_matrix(self, X: sp.spmatrix):
        """Filter out low-connectivity nodes from the matrix.
        
        Args:
            X: Original sparse matrix.
            
        Returns:
            M_filtered: Filtered sparse matrix.
            country_mask: Boolean array indicating which countries were kept.
            product_mask: Boolean array indicating which products were kept.
        """
        n_countries, n_products = X.shape
        M_filtered = X.copy()
        
        # Initialize masks (all True initially)
        country_mask = np.ones(n_countries, dtype=bool)
        product_mask = np.ones(n_products, dtype=bool)
        
        if self.iterative_filter:
            # Iteratively filter until matrix size stabilizes
            prev_shape = (-1, -1)
            iteration = 0
            while M_filtered.shape != prev_shape:
                prev_shape = M_filtered.shape
                iteration += 1
                
                # Filter low-diversification countries (rows)
                diversification = np.asarray(M_filtered.sum(axis=1)).ravel()
                valid_countries = diversification >= self.min_diversification
                
                # Update global mask
                temp_indices = np.where(country_mask)[0]
                country_mask[temp_indices[~valid_countries]] = False
                
                M_filtered = M_filtered[valid_countries, :]
                
                # Filter low-ubiquity products (columns)
                ubiquity = np.asarray(M_filtered.sum(axis=0)).ravel()
                valid_products = ubiquity >= self.min_ubiquity
                
                # Update global mask
                temp_indices = np.where(product_mask)[0]
                product_mask[temp_indices[~valid_products]] = False
                
                M_filtered = M_filtered[:, valid_products]
                
                if iteration > 100:  # Safety check
                    raise RuntimeError("Filtering did not stabilize after 100 iterations")
        else:
            # Single-pass filtering
            diversification = np.asarray(M_filtered.sum(axis=1)).ravel()
            valid_countries = diversification >= self.min_diversification
            country_mask = valid_countries
            M_filtered = M_filtered[valid_countries, :]
            
            ubiquity = np.asarray(M_filtered.sum(axis=0)).ravel()
            valid_products = ubiquity >= self.min_ubiquity
            product_mask = valid_products
            M_filtered = M_filtered[:, valid_products]
        
        return M_filtered, country_mask, product_mask
    
    def _expand_to_original(self, values_filtered: np.ndarray, mask: np.ndarray, n_orig: int):
        """Expand filtered values back to original dimensions with NaN for filtered nodes.
        
        Args:
            values_filtered: Values for the filtered subset.
            mask: Boolean mask indicating which original indices were kept.
            n_orig: Original number of elements.
            
        Returns:
            Array of length n_orig with NaN for filtered elements.
        """
        result = np.full(n_orig, np.nan, dtype=float)
        result[mask] = values_filtered
        return result

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
