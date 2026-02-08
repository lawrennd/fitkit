"""
Helper functions for community detection and within-community analysis.

These functions detect communities using eigenvector analysis and then
analyze ECI vs Fitness correlations within each community separately.
"""

import numpy as np
from scipy import sparse
from scipy.stats import pearsonr
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans


def detect_communities_from_eigenvectors(M, n_communities='auto', max_communities=5):
    """
    Detect communities using multiple eigenvectors of the transition matrix.
    
    Uses k-means clustering on the first k eigenvectors to identify communities.
    If n_communities='auto', uses the eigengap heuristic to choose k.
    
    Args:
        M: Binary incidence matrix (numpy array)
        n_communities: Number of communities or 'auto' for eigengap heuristic
        max_communities: Maximum number of communities to consider
    
    Returns:
        labels: Community assignment for each country (numpy array)
        n_communities: Number of communities detected
    """
    M = M.astype(float)
    n_countries, n_products = M.shape
    
    if n_countries < 10:
        return np.zeros(n_countries, dtype=int), 1
    
    k_c = M.sum(axis=1) + 1e-10
    k_p = M.sum(axis=0) + 1e-10
    
    D_c_inv = sparse.diags(1.0 / k_c)
    D_p_inv = sparse.diags(1.0 / k_p)
    M_sparse = sparse.csr_matrix(M)
    
    T_countries = D_c_inv @ M_sparse @ D_p_inv @ M_sparse.T
    
    # Compute more eigenvectors for community detection
    n_eigs = min(max_communities + 2, n_countries - 1)
    if n_eigs < 2:
        return np.zeros(n_countries, dtype=int), 1
    
    eigenvalues, eigenvectors = eigs(T_countries, k=n_eigs, which='LM')
    idx = np.argsort(np.real(eigenvalues))[::-1]
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:, idx])
    
    # Determine number of communities
    if n_communities == 'auto':
        # Use eigengap heuristic: look for largest gap in eigenvalues
        # Convert to Laplacian eigenvalues for gap analysis
        eigenvalues_L = 1 - eigenvalues
        gaps = np.diff(eigenvalues_L[1:])  # Skip trivial eigenvalue
        
        if len(gaps) == 0:
            n_communities = 1
        else:
            # Find largest gap (but cap at max_communities)
            n_communities = min(np.argmax(gaps) + 2, max_communities)
            
            # Only use multiple communities if gap is significant
            if n_communities > 1 and gaps[n_communities-2] < eigenvalues_L[1] * 0.1:
                n_communities = 1
    
    if n_communities == 1:
        return np.zeros(n_countries, dtype=int), 1
    
    # Use first k eigenvectors (skip first trivial one) for clustering
    embedding = eigenvectors[:, 1:n_communities]
    
    # K-means on eigenvector embedding
    kmeans = KMeans(n_clusters=n_communities, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embedding)
    
    return labels, n_communities


def analyze_within_communities(M, community_labels, ECI, FitnessComplexity):
    """
    Analyze ECI vs Fitness correlation within each detected community.
    
    Args:
        M: Binary incidence matrix
        community_labels: Community assignment for each country
        ECI: ECI estimator class
        FitnessComplexity: FitnessComplexity estimator class
    
    Returns:
        community_stats: List of dicts with per-community statistics, or None
    """
    n_communities = len(np.unique(community_labels))
    
    if n_communities == 1:
        return None
    
    M_sparse = sparse.csr_matrix(M)
    community_stats = []
    
    for comm_id in range(n_communities):
        comm_mask = community_labels == comm_id
        n_members = comm_mask.sum()
        
        if n_members < 3:
            continue
        
        # Extract subnetwork for this community
        M_comm = M[comm_mask, :]
        # Keep only products that appear in this community
        product_mask = M_comm.sum(axis=0) > 0
        M_comm = M_comm[:, product_mask]
        
        if M_comm.sum() < 10:  # Too sparse
            continue
        
        M_comm_sparse = sparse.csr_matrix(M_comm)
        
        try:
            # Compute ECI and Fitness within community
            eci_est = ECI()
            eci_comm, _ = eci_est.fit_transform(M_comm_sparse)
            
            fc_est = FitnessComplexity(n_iter=200, tol=1e-10, verbose=False)
            fitness_comm, _ = fc_est.fit_transform(M_comm_sparse)
            
            # Standardize
            fitness_comm_std = (fitness_comm - fitness_comm.mean()) / (fitness_comm.std() + 1e-10)
            
            # Compute correlation (remove NaNs)
            valid = ~np.isnan(eci_comm)
            if valid.sum() > 2:
                r, p = pearsonr(eci_comm[valid], fitness_comm_std[valid])
                
                community_stats.append({
                    'community_id': comm_id,
                    'n_members': n_members,
                    'correlation': r,
                    'p_value': p,
                    'eci': eci_comm,
                    'fitness': fitness_comm_std
                })
        except Exception as e:
            print(f"  Warning: Could not analyze community {comm_id}: {e}")
            continue
    
    return community_stats if community_stats else None
