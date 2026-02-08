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
        affinity: Literal["bipartite", "precomputed"] = "bipartite",
    ):
        self.method = method
        self.n_communities = n_communities
        self.max_communities = max_communities
        self.lambda_elongation = lambda_elongation
        self.random_state = random_state
        self.affinity = affinity
        
        # Attributes set during fitting
        self.labels_ = None
        self.n_communities_ = None
        self.eigenvalues_ = None
        self.n_iterations_ = None
    
    def _compute_eigenvectors(self, X):
        """Compute eigenvectors of transition/affinity matrix.
        
        Two modes depending on self.affinity:
        - 'bipartite': Compute T = D_c^{-1} M D_p^{-1} M^T (country-product)
        - 'precomputed': Use X directly as affinity/Laplacian matrix
        
        Parameters
        ----------
        X : sparse array
            Either bipartite adjacency matrix or pre-computed affinity matrix.
        
        Returns
        -------
        eigenvalues : ndarray
            Eigenvalues sorted in descending order.
        eigenvectors : ndarray
            Corresponding eigenvectors (columns).
        """
        if self.affinity == 'bipartite':
            # Bipartite network: compute transition matrix
            n_samples, n_features = X.shape
            
            # Compute degree sequences
            k_c = np.array(X.sum(axis=1)).ravel() + 1e-10
            k_p = np.array(X.sum(axis=0)).ravel() + 1e-10
            
            # Degree-normalized transition matrix
            D_c_inv = sparse.diags(1.0 / k_c)
            D_p_inv = sparse.diags(1.0 / k_p)
            
            # T = D_c^{-1} M D_p^{-1} M^T (country-country transitions)
            T = D_c_inv @ X @ D_p_inv @ X.T
            T_matrix = T
        else:
            # Pre-computed affinity/Laplacian: use directly
            if not sparse.issparse(X):
                X = sparse.csr_matrix(X)
            T_matrix = X
        
        # Use sparse eigensolver for stability on large matrices
        from scipy.sparse.linalg import eigsh
        n = T_matrix.shape[0]
        k_eigs = min(self.max_communities + 5, n - 2)
        
        try:
            # eigsh for sparse symmetric matrices (most stable)
            eigenvalues, eigenvectors = eigsh(T_matrix, k=k_eigs, which='LA')
            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        except Exception as e:
            # Fallback to dense for very small matrices
            T_dense = T_matrix.toarray() if sparse.issparse(T_matrix) else T_matrix
            if not np.all(np.isfinite(T_dense)):
                raise ValueError(f"Matrix contains NaN/Inf: {e}")
            eigenvalues, eigenvectors = np.linalg.eigh(T_dense)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        
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
        prev_centers = None  # For warm-starting
        prev_embedding = None  # Previous embedding for warm-start
        iteration = 0  # Initialize iteration counter
        
        for iteration in range(max_q - 1):
            # Extract q eigenvectors (MATLAB includes trivial eigenvector)
            # Use eigenvectors 0 to q-1 (Python 0-indexed)
            embedding = eigenvectors[:, 0:q]
            
            # Run elongated k-means with q clusters + 1 origin detector
            # Warm-start from previous iteration's centers if available
            labels_temp, origin_empty, centers = self._elongated_kmeans_with_origin(
                embedding, q, initial_centers=prev_centers, prev_embedding=prev_embedding
            )
            
            if origin_empty:
                # Origin cluster is empty -> found correct number
                # Re-run without origin to get final labels
                labels = self._elongated_kmeans(embedding, q, initial_centers=centers[:q])
                n_communities = q
                break
            
            # Origin captured points -> need more eigenvectors
            # Save centers and embedding for warm-start
            prev_centers = centers[:q]  # Exclude origin
            prev_embedding = embedding
            
            q += 1
            
            if q > max_q:
                # Reached max, use current q
                labels = self._elongated_kmeans(embedding, q - 1, initial_centers=prev_centers)
                n_communities = q - 1
                break
        
        if labels is None:
            # Fallback: single community
            labels = np.zeros(n_samples, dtype=int)
            n_communities = 1
        
        return labels, n_communities, iteration + 1
    
    def _elongated_kmeans_with_origin(self, X, k, initial_centers=None, prev_embedding=None):
        """Run elongated k-means with k clusters + 1 origin detector.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, q)
            Eigenvector embedding.
        k : int
            Number of clusters (excluding origin).
        initial_centers : ndarray or None
            Optional warm-start centers from previous iteration (shape k-1, q-1).
        prev_embedding : ndarray or None
            Previous eigenvector embedding (shape n_samples, q-1) for warm-start.
        
        Returns
        -------
        labels : ndarray
            Cluster assignments (0 to k, where k is origin).
        origin_empty : bool
            True if origin cluster captured no points.
        centers : ndarray
            Final cluster centers (for warm-starting next iteration).
        """
        n_samples, q = X.shape
        
        # Initialize k+1 centers (k clusters + origin)
        if (initial_centers is not None and prev_embedding is not None 
            and initial_centers.shape[0] == k - 1):
            # Warm-start: find closest points to old centers and use their new coordinates
            centers = self._extend_centers(X, initial_centers, k, prev_embedding)
        else:
            # Cold-start: initialize from scratch (k non-origin + 1 origin)
            centers = self._initialize_centers(X, k, include_origin=True)
        
        # Elongated k-means iterations
        max_iter = 100
        tol = 1e-4
        
        for _ in range(max_iter):
            # Assignment step
            # Use elongated distance metric
            labels = self._assign_to_nearest_center(X, centers)
            
            # Update step (all centers including origin can move)
            centers_new = np.zeros_like(centers)
            for i in range(k + 1):
                mask = labels == i
                if mask.sum() > 0:
                    centers_new[i] = X[mask].mean(axis=0)
                else:
                    # Empty cluster - keep previous position
                    centers_new[i] = centers[i]
            
            # Check convergence
            if np.linalg.norm(centers_new - centers) < tol:
                break
            
            centers = centers_new
        
        # Check if origin cluster is empty (exactly zero points)
        # This is the key test from the 2005 paper - origin captures points
        # if and only if there's an unaccounted cluster in the radial structure
        origin_empty = (labels == k).sum() == 0
        
        return labels, origin_empty, centers
    
    def _elongated_kmeans(self, X, k, initial_centers=None):
        """Run elongated k-means with k clusters (no origin).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, q)
            Eigenvector embedding.
        k : int
            Number of clusters.
        initial_centers : ndarray or None
            Optional initial centers (shape k, q) - excludes origin.
        
        Returns
        -------
        labels : ndarray
            Cluster assignments (0 to k-1).
        """
        n_samples, q = X.shape
        
        # Initialize k centers
        if initial_centers is not None and initial_centers.shape[0] == k:
            # Use provided centers (already excludes origin)
            centers = initial_centers.copy()
        else:
            # Initialize from scratch
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
    
    def _extend_centers(self, X, prev_centers, k, prev_embedding):
        """Extend centers using warm-start from previous iteration.
        
        Following MATLAB approach:
        1. Find which point was closest to each old center (in old embedding)
        2. Use those points' coordinates in the NEW embedding as initial centers
        3. Add origin as final center
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, q)
            Current eigenvector embedding (q dimensions).
        prev_centers : ndarray of shape (k-1, q-1)
            Centers from previous iteration (q-1 dimensions, excluding origin).
        k : int
            Number of non-origin clusters for this iteration.
        prev_embedding : ndarray of shape (n_samples, q-1)
            Previous eigenvector embedding.
        
        Returns
        -------
        centers : ndarray of shape (k+1, q)
            Extended centers including origin.
        """
        n_samples, q = X.shape
        
        # Find point closest to each previous center (in old embedding space)
        closest_points = []
        n_prev_centers = prev_centers.shape[0]
        for i in range(n_prev_centers):
            distances = np.linalg.norm(prev_embedding - prev_centers[i], axis=1)
            closest_idx = np.argmin(distances)
            closest_points.append(closest_idx)
        
        # Use those points' coordinates in NEW embedding
        new_centers = X[closest_points]
        
        # Add a NEW k-th center (for the additional cluster)
        # Find point that is far from existing centers in new embedding
        norms = np.linalg.norm(X, axis=1)
        norms_sq = norms ** 2
        
        min_score = np.inf
        best_idx = None
        for i in range(n_samples):
            if i in closest_points:
                continue
            # Compute projection ratio onto existing centers
            S = 0.0
            for center in new_centers:
                projection = np.dot(X[i], center)
                S += (projection ** 2) / (norms_sq[i] + 1e-10)
            
            if S < min_score:
                min_score = S
                best_idx = i
        
        if best_idx is None:
            best_idx = np.random.randint(n_samples)
        
        new_kth_center = X[best_idx].reshape(1, -1)
        
        # Combine: n_prev_centers old centers + 1 new k-th center + origin
        centers = np.vstack([new_centers, new_kth_center, np.zeros((1, q))])
        
        return centers
    
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
        norms_sq = norms ** 2 + 1e-10  # Avoid division by zero
        
        # MATLAB-style initialization (only for k=2 initially)
        # First center: farthest from origin
        first_idx = np.argmax(norms)
        
        if k == 1:
            centers = X[first_idx:first_idx+1]
        elif k == 2:
            # Second center: minimize S = (projection onto first)^2 / norm^2
            # This finds the point most perpendicular to the first center
            projections = np.dot(X, X[first_idx])
            S = (projections ** 2) / norms_sq
            second_idx = np.argmin(S)
            centers = np.array([X[first_idx], X[second_idx]])
        else:
            # For k > 2, use iterative selection (minimizing accumulated projections)
            selected = [first_idx]
            
            for _ in range(k - 1):
                min_score = np.inf
                best_idx = None
                
                for i in range(n_samples):
                    if i in selected:
                        continue
                    
                    # S = sum of (projection onto each existing center)^2 / norm^2
                    S = 0.0
                    for j in selected:
                        projection = np.dot(X[i], X[j])
                        S += (projection ** 2) / norms_sq[i]
                    
                    if S < min_score:
                        min_score = S
                        best_idx = i
                
                if best_idx is not None:
                    selected.append(best_idx)
                else:
                    selected.append(np.random.randint(n_samples))
            
            centers = X[selected]
        
        if include_origin:
            # Add origin as final center
            origin = np.zeros((1, q))
            centers = np.vstack([centers, origin])
        
        return centers
    
    def _assign_to_nearest_center_euclidean(self, X, centers):
        """Assign points to nearest center using standard Euclidean distance.
        
        Used for disconnected components where eigenvectors are block indicators.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, q)
            Points in eigenvector embedding.
        centers : ndarray of shape (k, q)
            Cluster centers (including origin as last center).
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster assignments.
        """
        n_samples = X.shape[0]
        k = centers.shape[0]
        
        # Compute Euclidean distances to all centers
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(X - centers[i], axis=1)
        
        # Assign to nearest
        labels = np.argmin(distances, axis=1)
        
        return labels
    
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
        radial_coeff = (diff @ c_hat).reshape(-1, 1)  # Shape (n_samples, 1)
        radial_proj = radial_coeff * c_hat  # Broadcasting to (n_samples, q)
        
        # Tangential component
        tangential = diff - radial_proj
        
        # Elongated distance: downweight radial, penalize tangential
        # For λ < 1: d² = λ * ||radial||² + (1/λ) * ||tangential||²
        lam = self.lambda_elongation
        
        radial_dist_sq = np.sum(radial_proj ** 2, axis=1)
        tangential_dist_sq = np.sum(tangential ** 2, axis=1)
        
        distances_sq = lam * radial_dist_sq + (1.0 / lam) * tangential_dist_sq
        
        return np.sqrt(distances_sq)
    
    def fit(self, X, eigenvectors=None):
        """Detect communities in the network.
        
        Parameters
        ----------
        X : sparse or dense array
            Input matrix (ignored if eigenvectors provided):
            - If affinity='bipartite': Bipartite adjacency (n_samples, n_features)
            - If affinity='precomputed': Normalized Laplacian (n_samples, n_samples)
        eigenvectors : ndarray of shape (n_samples, n_eigenvectors), optional
            Pre-computed eigenvectors to use directly.
            Columns should be sorted by eigenvalue magnitude (descending).
            If provided, X is ignored and eigendecomposition is skipped.
        
        Returns
        -------
        self : CommunityDetector
            Fitted estimator.
        """
        if self.method != "iterative":
            raise ValueError(f"Unknown method: {self.method}. Only 'iterative' is supported.")
        
        # Handle pre-computed eigenvectors
        if eigenvectors is not None:
            eigenvalues = None
            n_samples = eigenvectors.shape[0]
        else:
            # Convert to sparse if needed
            if not sparse.issparse(X):
                X = sparse.csr_matrix(X)
            
            n_samples = X.shape[0]
            
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
            
            # Use first n_communities eigenvectors
            embedding = eigenvectors[:, 0:n_communities]
            labels = self._elongated_kmeans(embedding, n_communities)
            n_iters = 1
        
        self.labels_ = labels
        self.n_communities_ = n_communities
        self.n_iterations_ = n_iters
        
        return self
    
    def fit_predict(self, X, eigenvectors=None):
        """Detect communities and return labels.
        
        Parameters
        ----------
        X : sparse or dense array
            Input matrix (see fit() for details).
        eigenvectors : ndarray, optional
            Pre-computed eigenvectors (see fit() for details).
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Community labels for each node.
        """
        self.fit(X, eigenvectors=eigenvectors)
        return self.labels_
