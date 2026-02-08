"""Community detection using iterative eigenvector algorithm.

This module implements the CommunityDetector class, which uses an iterative
approach to automatically determine the number of communities in a bipartite
network using spectral methods.

The algorithm is based on Sanguinetti, Lawrence & Laidler (2005) and uses:
- Eigenvectors of the graph Laplacian as embedding coordinates
- Elongated k-means clustering that exploits radial structure
- Origin-detector validation to determine intrinsic dimensionality

Theoretical justification from diffusion maps framework (economic-fitness.tex):
- Eigenvectors represent geometric modes of diffusion process
- Spectral gap measures separation timescale between communities
- Radial elongation occurs when projection dimension is insufficient
- Iterative addition of eigenvectors finds intrinsic dimensionality

References:
    Sanguinetti, G., Laidler, J., & Lawrence, N. D. (2005).
    "Automatic determination of the number of clusters using spectral algorithms".
    IEEE MLSP.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from typing import Literal, Optional, Union


class CommunityDetector:
    """Detect communities in bipartite networks using iterative eigenvector method.
    
    This class implements the algorithm from Sanguinetti, Lawrence & Laidler (2005),
    which automatically determines the number of communities by iteratively adding
    eigenvectors until an origin-detector cluster remains empty.
    
    The algorithm uses elongated k-means with a Mahalanobis-like distance metric
    that accounts for radial elongation in eigenvector space, making it particularly
    effective for networks with hierarchical or modular structure.
    
    Parameters
    ----------
    method : {'iterative'}, default='iterative'
        Community detection method. Currently only 'iterative' is supported.
    n_communities : int or 'auto', default='auto'
        Number of communities to detect. If 'auto', uses iterative algorithm
        to determine automatically.
    max_communities : int, default=8
        Maximum number of communities to consider when n_communities='auto'.
    lambda_elongation : float, default=0.2
        Elongation parameter for elongated k-means (0 < lambda < 1).
        Smaller values give stronger radial elongation.
    random_state : int or None, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Community labels for each node (0 to n_communities_-1).
    n_communities_ : int
        Number of communities detected.
    eigenvalues_ : ndarray
        Eigenvalues of the graph Laplacian (for diagnostics).
    n_iterations_ : int
        Number of iterations until convergence (when using 'auto').
    
    Examples
    --------
    >>> from fitkit.community import CommunityDetector
    >>> detector = CommunityDetector(n_communities='auto')
    >>> labels = detector.fit_predict(M)
    >>> print(f"Found {detector.n_communities_} communities")
    
    Notes
    -----
    The elongated k-means distance metric is:
    
    .. math::
        d^2(x, c) = (x-c)^T M (x-c)
    
    where M downweights distances along the radial direction:
    
    .. math::
        M = \\frac{1}{\\lambda}(I - \\frac{cc^T}{||c||^2}) + \\lambda\\frac{cc^T}{||c||^2}
    
    This exploits the geometric fact that insufficient-dimensional projections
    produce radially elongated clusters.
    
    References
    ----------
    .. [1] Sanguinetti, G., Laidler, J., & Lawrence, N. D. (2005).
           "Automatic determination of the number of clusters using spectral algorithms".
           IEEE MLSP.
    """
    
    def __init__(
        self,
        method: Literal["iterative"] = "iterative",
        n_communities: Union[int, Literal["auto"]] = "auto",
        max_communities: int = 8,
        lambda_elongation: float = 0.2,
        random_state: Optional[int] = None,
    ):
        self.method = method
        self.n_communities = n_communities
        self.max_communities = max_communities
        self.lambda_elongation = lambda_elongation
        self.random_state = random_state
        
        # Attributes set during fitting
        self.labels_ = None
        self.n_communities_ = None
        self.eigenvalues_ = None
        self.n_iterations_ = None
    
    def _compute_eigenvectors(self, X):
        """Compute eigenvectors of transition matrix for bipartite network.
        
        Parameters
        ----------
        X : sparse array
            Bipartite adjacency matrix.
        
        Returns
        -------
        eigenvalues : ndarray
            Eigenvalues sorted in descending order.
        eigenvectors : ndarray
            Corresponding eigenvectors (columns).
        """
        n_samples, n_features = X.shape
        
        # Compute degree sequences
        k_c = np.array(X.sum(axis=1)).ravel() + 1e-10
        k_p = np.array(X.sum(axis=0)).ravel() + 1e-10
        
        # Degree-normalized transition matrix
        D_c_inv = sparse.diags(1.0 / k_c)
        D_p_inv = sparse.diags(1.0 / k_p)
        
        # T = D_c^{-1} M D_p^{-1} M^T (country-country transitions)
        T = D_c_inv @ X @ D_p_inv @ X.T
        
        # Compute eigenvectors
        # Need to ensure k < n_samples - 1 for sparse eigs
        n_eigs = min(self.max_communities + 2, n_samples - 2)
        n_eigs = max(2, n_eigs)  # At least 2 eigenvectors
        
        # For very small networks, use dense eigendecomposition
        if n_eigs >= n_samples - 2 or n_samples < 10:
            from scipy.linalg import eigh
            T_dense = T.toarray() if sparse.issparse(T) else T
            eigenvalues, eigenvectors = eigh(T_dense)
            # Sort by magnitude (descending)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        else:
            eigenvalues, eigenvectors = eigs(T, k=n_eigs, which='LM')
            # Sort by magnitude (descending)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = np.real(eigenvalues[idx])
            eigenvectors = np.real(eigenvectors[:, idx])
        
        return eigenvalues, eigenvectors
    
    def _iterative_detection(self, eigenvectors, eigenvalues):
        """Iterative algorithm to determine number of communities.
        
        Parameters
        ----------
        eigenvectors : ndarray
            Eigenvectors of transition matrix.
        eigenvalues : ndarray
            Corresponding eigenvalues.
        
        Returns
        -------
        labels : ndarray
            Community assignments.
        n_communities : int
            Number of communities detected.
        n_iterations : int
            Number of iterations performed.
        """
        n_samples = eigenvectors.shape[0]
        
        # Start with q=2 eigenvectors (skip trivial one)
        q = 2
        max_q = min(self.max_communities, eigenvectors.shape[1] - 1)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        labels = None
        n_communities = 1
        
        for iteration in range(max_q - 1):
            # Extract q eigenvectors (skip first trivial one)
            embedding = eigenvectors[:, 1:q+1]
            
            # Run elongated k-means with q clusters + 1 origin detector
            labels_temp, origin_empty = self._elongated_kmeans_with_origin(
                embedding, q
            )
            
            if origin_empty:
                # Origin cluster is empty -> found correct number
                # Re-run without origin to get final labels
                labels = self._elongated_kmeans(embedding, q)
                n_communities = q
                break
            
            # Origin captured points -> need more eigenvectors
            q += 1
            
            if q > max_q:
                # Reached max, use current q
                labels = self._elongated_kmeans(embedding, q - 1)
                n_communities = q - 1
                break
        
        if labels is None:
            # Fallback: single community
            labels = np.zeros(n_samples, dtype=int)
            n_communities = 1
        
        return labels, n_communities, iteration + 1
    
    def _elongated_kmeans_with_origin(self, X, k):
        """Run elongated k-means with k clusters + 1 origin detector.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, q)
            Eigenvector embedding.
        k : int
            Number of clusters (excluding origin).
        
        Returns
        -------
        labels : ndarray
            Cluster assignments (0 to k, where k is origin).
        origin_empty : bool
            True if origin cluster captured no points.
        """
        n_samples, q = X.shape
        
        # Initialize k+1 centers (k clusters + origin)
        centers = self._initialize_centers(X, k)
        
        # Elongated k-means iterations
        max_iter = 100
        tol = 1e-4
        
        for _ in range(max_iter):
            # Assignment step
            labels = self._assign_to_nearest_center(X, centers)
            
            # Update step
            centers_new = np.zeros_like(centers)
            for i in range(k + 1):
                mask = labels == i
                if mask.sum() > 0:
                    centers_new[i] = X[mask].mean(axis=0)
                else:
                    # Empty cluster - reinitialize randomly
                    centers_new[i] = X[np.random.randint(n_samples)]
            
            # Origin stays at origin
            centers_new[k] = np.zeros(q)
            
            # Check convergence
            if np.linalg.norm(centers_new - centers) < tol:
                break
            
            centers = centers_new
        
        # Check if origin cluster is empty (exactly zero points)
        # This is the key test from the 2005 paper - origin captures points
        # if and only if there's an unaccounted cluster in the radial structure
        origin_empty = (labels == k).sum() == 0
        
        return labels, origin_empty
    
    def _elongated_kmeans(self, X, k):
        """Run elongated k-means with k clusters (no origin).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, q)
            Eigenvector embedding.
        k : int
            Number of clusters.
        
        Returns
        -------
        labels : ndarray
            Cluster assignments (0 to k-1).
        """
        n_samples, q = X.shape
        
        # Initialize k centers
        centers = self._initialize_centers(X, k, include_origin=False)
        
        # Elongated k-means iterations
        max_iter = 100
        tol = 1e-4
        
        for _ in range(max_iter):
            # Assignment step
            labels = self._assign_to_nearest_center(X, centers, include_origin=False)
            
            # Update step
            centers_new = np.zeros_like(centers)
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    centers_new[i] = X[mask].mean(axis=0)
                else:
                    # Empty cluster - reinitialize randomly
                    centers_new[i] = X[np.random.randint(n_samples)]
            
            # Check convergence
            if np.linalg.norm(centers_new - centers) < tol:
                break
            
            centers = centers_new
        
        return labels
    
    def _initialize_centers(self, X, k, include_origin=True):
        """Initialize cluster centers.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, q)
            Eigenvector embedding.
        k : int
            Number of non-origin clusters.
        include_origin : bool
            Whether to include origin as (k+1)-th center.
        
        Returns
        -------
        centers : ndarray of shape (k+1, q) or (k, q)
            Initial cluster centers.
        """
        n_samples, q = X.shape
        norms = np.linalg.norm(X, axis=1)
        
        # First center: farthest from origin
        first_idx = np.argmax(norms)
        
        if k == 1:
            centers = X[first_idx:first_idx+1]
        else:
            # Subsequent centers: maximize distance while minimizing dot product
            # (trying to spread around the sphere)
            selected = [first_idx]
            
            for _ in range(k - 1):
                max_score = -np.inf
                best_idx = None
                
                for i in range(n_samples):
                    if i in selected:
                        continue
                    
                    # Score: norm minus similarity to existing centers
                    score = norms[i]
                    for j in selected:
                        similarity = np.abs(np.dot(X[i], X[j])) / (norms[i] * norms[j] + 1e-10)
                        score -= similarity
                    
                    if score > max_score:
                        max_score = score
                        best_idx = i
                
                if best_idx is not None:
                    selected.append(best_idx)
                else:
                    # Fallback: random point
                    selected.append(np.random.randint(n_samples))
            
            centers = X[selected]
        
        if include_origin:
            # Add origin as final center
            origin = np.zeros((1, q))
            centers = np.vstack([centers, origin])
        
        return centers
    
    def _assign_to_nearest_center(self, X, centers, include_origin=True):
        """Assign points to nearest center using elongated distance metric.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, q)
            Points in eigenvector embedding.
        centers : ndarray of shape (k, q)
            Cluster centers.
        include_origin : bool
            If True, last center is origin (use Euclidean distance).
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster assignments.
        """
        n_samples = X.shape[0]
        k = centers.shape[0]
        
        # Compute distances to all centers
        distances = np.zeros((n_samples, k))
        
        for i in range(k):
            if include_origin and i == k - 1:
                # Origin: use Euclidean distance
                distances[:, i] = np.linalg.norm(X - centers[i], axis=1)
            else:
                # Non-origin: use elongated distance
                distances[:, i] = self._compute_elongated_distance(
                    X, centers[i]
                )
        
        # Assign to nearest
        labels = np.argmin(distances, axis=1)
        
        return labels
    
    def _compute_elongated_distance(self, X, center):
        """Compute elongated Mahalanobis distance to center.
        
        The distance metric downweights radial direction and penalizes
        tangential direction:
        
        d²(x, c) = (x-c)ᵀ M (x-c)
        
        where M = (1/λ)(I - ccᵀ/||c||²) + λ(ccᵀ/||c||²)
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, q)
            Points.
        center : ndarray of shape (q,)
            Cluster center.
        
        Returns
        -------
        distances : ndarray of shape (n_samples,)
            Elongated distances.
        """
        diff = X - center
        
        center_norm = np.linalg.norm(center)
        if center_norm < 1e-10:
            # Center at origin: use Euclidean
            return np.linalg.norm(diff, axis=1)
        
        # Normalized center direction
        c_hat = center / center_norm
        
        # Projection onto radial direction
        radial_proj = np.outer(diff @ c_hat, c_hat)
        
        # Tangential component
        tangential = diff - radial_proj
        
        # Elongated distance: downweight radial, penalize tangential
        # d² = λ * ||radial||² + (1/λ) * ||tangential||²
        lam = self.lambda_elongation
        
        radial_dist_sq = np.sum(radial_proj ** 2, axis=1)
        tangential_dist_sq = np.sum(tangential ** 2, axis=1)
        
        distances_sq = lam * radial_dist_sq + (1.0 / lam) * tangential_dist_sq
        
        return np.sqrt(distances_sq)
    
    def fit(self, X):
        """Detect communities in the network.
        
        Parameters
        ----------
        X : sparse or dense array of shape (n_samples, n_features)
            Bipartite adjacency matrix.
        
        Returns
        -------
        self : CommunityDetector
            Fitted estimator.
        """
        if self.method != "iterative":
            raise ValueError(f"Unknown method: {self.method}. Only 'iterative' is supported.")
        
        # Convert to sparse if needed
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        
        n_samples, n_features = X.shape
        
        # Handle edge cases
        if n_samples < 3:
            # Too small for meaningful clustering
            self.labels_ = np.zeros(n_samples, dtype=int)
            self.n_communities_ = 1
            self.eigenvalues_ = np.array([1.0])
            self.n_iterations_ = 0
            return self
        
        # Compute eigenvectors
        eigenvalues, eigenvectors = self._compute_eigenvectors(X)
        self.eigenvalues_ = eigenvalues
        
        # Determine number of communities
        if self.n_communities == 'auto':
            labels, n_communities, n_iters = self._iterative_detection(
                eigenvectors, eigenvalues
            )
        else:
            # Fixed number of communities
            n_communities = int(self.n_communities)
            if n_communities < 1 or n_communities > n_samples:
                raise ValueError(
                    f"n_communities must be between 1 and {n_samples}, got {n_communities}"
                )
            
            # Use first n_communities eigenvectors (skip trivial one)
            embedding = eigenvectors[:, 1:n_communities+1]
            labels = self._elongated_kmeans(embedding, n_communities)
            n_iters = 1
        
        self.labels_ = labels
        self.n_communities_ = n_communities
        self.n_iterations_ = n_iters
        
        return self
    
    def fit_predict(self, X):
        """Detect communities and return labels.
        
        Parameters
        ----------
        X : sparse or dense array of shape (n_samples, n_features)
            Bipartite adjacency matrix.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Community labels for each node.
        """
        self.fit(X)
        return self.labels_
