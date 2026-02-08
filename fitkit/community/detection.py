"""
Community detection using spectral clustering.

This module provides a simple wrapper around the SpectralCluster implementation
for detecting communities in bipartite networks (e.g., country-product networks).
"""

import numpy as np
from scipy import sparse

from .cluster import SpectralCluster
from .affinity import normalize_laplacian


def build_bipartite_transition_matrix(M):
    """
    Build transition matrix for bipartite network.
    
    Following the formulation in economic-fitness.tex:
    T = D_c^{-1} M D_p^{-1} M^T
    
    where:
    - M is the bipartite adjacency matrix (countries × products)
    - D_c is diagonal matrix of country degrees
    - D_p is diagonal matrix of product degrees
    
    Parameters
    ----------
    M : array-like of shape (n_countries, n_products)
        Bipartite adjacency matrix (can be sparse or dense)
        
    Returns
    -------
    T : ndarray of shape (n_countries, n_countries)
        Transition matrix (dense)
    """
    # Convert to dense if sparse
    if sparse.issparse(M):
        M = M.toarray()
    
    # Compute degree matrices
    # D_c = row sums of M (country degrees)
    # D_p = column sums of M (product degrees)
    D_c = M.sum(axis=1)  # shape (n_countries,)
    D_p = M.sum(axis=0)  # shape (n_products,)
    
    # Avoid division by zero
    D_c = np.where(D_c > 0, D_c, 1.0)
    D_p = np.where(D_p > 0, D_p, 1.0)
    
    # Compute T = D_c^{-1} M D_p^{-1} M^T
    # Step 1: D_p^{-1} M^T = (M D_p^{-1})^T
    M_weighted = M / D_p[np.newaxis, :]  # Broadcasting: M * diag(1/D_p)
    
    # Step 2: M * (D_p^{-1} M^T) = M * M_weighted^T
    MM_T = M @ M_weighted.T
    
    # Step 3: D_c^{-1} * (M D_p^{-1} M^T)
    T = MM_T / D_c[:, np.newaxis]  # Broadcasting: diag(1/D_c) * MM_T
    
    return T


class CommunityDetector:
    """
    Detect communities in bipartite networks using spectral clustering.
    
    This is a wrapper around SpectralCluster that handles bipartite network
    input (e.g., country-product matrices) by first computing the appropriate
    transition matrix.
    
    Parameters
    ----------
    lambda_elongation : float, default=0.2
        Elongation parameter for the distance metric in k-means.
        Smaller values create more elongated clusters.
    max_communities : int, default=10
        Maximum number of communities to detect.
    random_state : int or None, default=None
        Random seed for reproducibility.
    affinity : str, default='bipartite'
        Type of input:
        - 'bipartite': M is a bipartite adjacency matrix
        - 'precomputed': M is a pre-computed normalized Laplacian
        
    Attributes
    ----------
    labels_ : ndarray
        Community labels for each node
    n_communities_ : int
        Number of communities detected
    cluster_ : SpectralCluster
        The underlying spectral clustering object
        
    Examples
    --------
    >>> import numpy as np
    >>> from fitkit.community import CommunityDetector
    >>> 
    >>> # Create a simple bipartite network
    >>> M = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> detector = CommunityDetector(max_communities=5)
    >>> detector.fit(M)
    >>> print(f"Found {detector.n_communities_} communities")
    """
    
    def __init__(self, lambda_elongation=0.2, max_communities=10, 
                 random_state=None, affinity='bipartite'):
        self.lambda_elongation = lambda_elongation
        self.max_communities = max_communities
        self.random_state = random_state
        self.affinity = affinity
        
    def fit(self, X, eigenvectors=None):
        """
        Fit the community detector.
        
        Parameters
        ----------
        X : array-like
            If affinity='bipartite': shape (n_countries, n_products), bipartite matrix
            If affinity='precomputed': shape (n, n), normalized Laplacian matrix
        eigenvectors : ndarray or None
            If provided, use these eigenvectors directly instead of computing
            from X. Shape should be (n_samples, n_eigenvectors).
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        if eigenvectors is not None:
            # Direct eigenvector input - use SpectralCluster's internal fitting
            raise NotImplementedError(
                "Direct eigenvector input not yet supported. "
                "Use affinity='precomputed' and pass normalized Laplacian instead."
            )
        
        if self.affinity == 'bipartite':
            # Build transition matrix from bipartite network
            T = build_bipartite_transition_matrix(X)
            # T is already a form of normalized matrix, but we need to treat it
            # as an affinity matrix for SpectralCluster
            affinity_matrix = T
        elif self.affinity == 'precomputed':
            # X is already a normalized Laplacian or affinity matrix
            affinity_matrix = X
            if sparse.issparse(affinity_matrix):
                affinity_matrix = affinity_matrix.toarray()
        else:
            raise ValueError(f"Unknown affinity type: {self.affinity}")
        
        # Create SpectralCluster - but we need to bypass its affinity computation
        # We'll create a custom version that uses precomputed Laplacian
        from scipy import linalg
        from scipy.sparse.linalg import eigsh
        
        # Compute eigenvectors directly using sparse solver for stability
        n = affinity_matrix.shape[0]
        k_eigs = min(self.max_communities + 5, n - 2)
        
        try:
            # Use sparse eigensolver for large symmetric matrices
            affinity_sparse = sparse.csr_matrix(affinity_matrix)
            eigvals, eigvecs = eigsh(affinity_sparse, k=k_eigs, which='LA')
        except Exception:
            # Fallback to dense for small matrices
            eigvals, eigvecs = linalg.eigh(affinity_matrix)
        
        # Sort by eigenvalue in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Now run the clustering algorithm from SpectralCluster
        # We'll create a SpectralCluster instance but override its eigendecomposition
        n_samples = affinity_matrix.shape[0]
        all_eigvecs = eigvecs[:, :self.max_communities]
        eigenvalues = eigvals[:self.max_communities]
        
        # Import ElongatedKMeans
        from .kmeans import ElongatedKMeans
        
        # Implement the iterative algorithm (matching cluster.py lines 166-276)
        Dim = 2
        ExtraCluster = False
        
        # Initialize with first Dim eigenvectors
        PcEig = all_eigvecs[:, :Dim]
        
        # Initialize first two centers
        norms = np.sum(PcEig ** 2, axis=1)
        idx_first = np.argmax(norms)
        centers = PcEig[[idx_first], :]
        
        # Second center: minimize (projection on first)^2 / norm
        projections_sq = (PcEig @ centers[0]) ** 2
        S = projections_sq / (norms + 1e-10)
        idx_second = np.argmin(S)
        centers = PcEig[[idx_first, idx_second], :]
        
        # Main loop
        kmeans = None
        
        while not ExtraCluster and Dim < self.max_communities:
            # Add origin as (Dim+1)-th center
            centers_with_origin = np.vstack([centers, np.zeros(Dim)])
            
            # Run elongated k-means
            kmeans = ElongatedKMeans(
                n_clusters=Dim + 1,
                lambda_=self.lambda_elongation,
                epsilon=0.0001,
                max_iter=100,
                tol=1e-4
            )
            kmeans.fit(PcEig, centers_with_origin)
            
            # Check if any points assigned to origin (last cluster)
            n_points_in_origin = np.sum(kmeans.labels_ == Dim)
            
            if n_points_in_origin > 0:
                # There's an extra cluster - expand dimensionality
                Dim += 1
                
                if Dim >= self.max_communities:
                    # Hit max clusters - use current clustering without origin
                    centers_final = kmeans.cluster_centers_[:-1]
                    n_final_clusters = len(centers_final)
                    kmeans_final = ElongatedKMeans(
                        n_clusters=n_final_clusters,
                        lambda_=self.lambda_elongation,
                        epsilon=0.0001,
                        max_iter=100,
                        tol=1e-4
                    )
                    kmeans_final.fit(PcEig, centers_final)
                    kmeans = kmeans_final
                    centers = kmeans.cluster_centers_
                    Dim = n_final_clusters
                    break
                
                # Take next eigenvector
                PcEig = all_eigvecs[:, :Dim]
                
                # Re-initialize centers
                centers = np.zeros((Dim, Dim))
                for i in range(Dim):
                    if i < len(kmeans.cluster_centers_) - 1:  # Skip origin
                        old_center = kmeans.cluster_centers_[i]
                        old_center_padded = np.pad(old_center, (0, 1), mode='constant')
                        distances = np.sum((PcEig - old_center_padded) ** 2, axis=1)
                        closest_point_idx = np.argmin(distances)
                        centers[i] = PcEig[closest_point_idx]
                    else:
                        # For additional center, take furthest from existing
                        if i == 0:
                            centers[i] = PcEig[np.argmax(np.sum(PcEig ** 2, axis=1))]
                        else:
                            distances_to_all = np.zeros((n_samples, i))
                            for j in range(i):
                                distances_to_all[:, j] = np.sum((PcEig - centers[j]) ** 2, axis=1)
                            min_distances = np.min(distances_to_all, axis=1)
                            centers[i] = PcEig[np.argmax(min_distances)]
            else:
                # No points in origin cluster - found correct number
                ExtraCluster = True
                centers = kmeans.cluster_centers_[:-1]  # Remove origin
        
        # Store final results
        self.n_communities_ = Dim
        self.eigenvectors_ = PcEig
        self.centers_ = centers
        
        if kmeans is not None:
            self.labels_ = kmeans.labels_
        else:
            self.labels_ = np.zeros(n_samples, dtype=int)
        
        return self
    
    def fit_predict(self, X, eigenvectors=None):
        """
        Fit and return cluster labels.
        
        Parameters
        ----------
        X : array-like
            Input data (see fit())
        eigenvectors : ndarray or None
            Optional pre-computed eigenvectors
            
        Returns
        -------
        labels : ndarray
            Cluster labels
        """
        return self.fit(X, eigenvectors).labels_
