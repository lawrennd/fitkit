"""Tests for community detection module.

Based on reference MATLAB implementation at:
https://github.com/lawrennd/spectral/blob/master/matlab/
"""
import numpy as np
import pytest
from scipy import sparse

from fitkit.community.detection import CommunityDetector


def compute_gaussian_affinity(X, sigma2):
    """Compute Gaussian affinity matrix matching MATLAB demoCircles.m."""
    n = len(X)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = np.exp(-np.linalg.norm(X[i] - X[j])**2 / sigma2)
    return A


def affinity_to_normalized_laplacian(A):
    """Convert affinity matrix to normalized Laplacian."""
    D = A.sum(axis=1)
    # Handle zero-degree nodes
    D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    D_inv_sqrt = np.diag(D_inv_sqrt)
    L = D_inv_sqrt @ A @ D_inv_sqrt
    return L


class TestCommunityDetector:
    """Test suite for CommunityDetector class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        detector = CommunityDetector(max_communities=5, random_state=42)
        assert detector.max_communities == 5
        assert detector.random_state == 42
        assert detector.lambda_elongation == 0.2  # default
    
    def test_two_block_bipartite(self):
        """Test perfect 2-block bipartite network."""
        # Create perfect 2-block structure
        M = np.zeros((20, 16))
        M[0:10, 0:8] = 1
        M[10:20, 8:16] = 1
        M_sparse = sparse.csr_matrix(M)
        
        detector = CommunityDetector(max_communities=10, random_state=42)
        labels = detector.fit_predict(M_sparse)
        
        # Should detect exactly 2 communities
        assert detector.n_communities_ == 2
        
        # Check purity: all points in block 1 should have same label
        block1_labels = labels[0:10]
        block2_labels = labels[10:20]
        
        assert len(np.unique(block1_labels)) == 1
        assert len(np.unique(block2_labels)) == 1
        assert block1_labels[0] != block2_labels[0]
    
    def test_three_block_bipartite(self):
        """Test perfect 3-block bipartite network."""
        M = np.zeros((30, 24))
        M[0:10, 0:8] = 1
        M[10:20, 8:16] = 1
        M[20:30, 16:24] = 1
        M_sparse = sparse.csr_matrix(M)
        
        detector = CommunityDetector(max_communities=10, random_state=42)
        labels = detector.fit_predict(M_sparse)
        
        # Should detect exactly 3 communities
        assert detector.n_communities_ == 3
        
        # Check purity
        block1_labels = labels[0:10]
        block2_labels = labels[10:20]
        block3_labels = labels[20:30]
        
        assert len(np.unique(block1_labels)) == 1
        assert len(np.unique(block2_labels)) == 1
        assert len(np.unique(block3_labels)) == 1
        
        # All three blocks should have different labels
        unique_labels = {block1_labels[0], block2_labels[0], block3_labels[0]}
        assert len(unique_labels) == 3
    
    def test_modular_with_cross_connections(self):
        """Test modular structure with some cross-connections."""
        np.random.seed(42)
        
        # Start with 2-block
        M = np.zeros((20, 16))
        M[0:10, 0:8] = 1
        M[10:20, 8:16] = 1
        
        # Add cross-connections
        for i in range(10):
            products = np.random.choice(range(8, 16), size=1, replace=False)
            M[i, products] = 1
        for i in range(10, 20):
            products = np.random.choice(range(0, 8), size=1, replace=False)
            M[i, products] = 1
        
        M_sparse = sparse.csr_matrix(M)
        detector = CommunityDetector(max_communities=10, random_state=42)
        labels = detector.fit_predict(M_sparse)
        
        # Should still detect 2 main communities despite noise
        assert detector.n_communities_ == 2
    
    @pytest.mark.skip(reason="BLAS/LAPACK segfault on dense 300x300 matrices - environmental issue")
    def test_concentric_circles_gaussian_affinity(self):
        """Test three CONCENTRIC circles from MATLAB demoCircles.m.
        
        This is the challenging case where regular k-means fails because
        clusters are nested (radii 1, 2, 3) rather than spatially separated.
        """
        # Match MATLAB: randn('seed',1); rand('seed',1);
        np.random.seed(1)
        
        npts = 100
        step = 2*np.pi/npts
        theta = np.arange(step, 2*np.pi + step, step)
        
        # Random radius perturbations
        radius = np.random.randn(npts)
        
        # Three concentric circles: r=1, r=2, r=3 (with 0.1*noise)
        r1 = np.ones(npts) + 0.1*radius
        r2 = 2*np.ones(npts) + 0.1*radius  
        r3 = 3*np.ones(npts) + 0.1*radius
        
        # Generate points (all centered at origin)
        circle1 = np.column_stack([r1*np.cos(theta), r1*np.sin(theta)])
        circle2 = np.column_stack([r2*np.cos(theta), r2*np.sin(theta)])
        circle3 = np.column_stack([r3*np.cos(theta), r3*np.sin(theta)])
        
        X = np.vstack([circle1, circle2, circle3])
        true_labels = np.array([0]*npts + [1]*npts + [2]*npts)
        
        # Compute Gaussian affinity matrix (sigma2 = 0.05 as in MATLAB)
        A = compute_gaussian_affinity(X, sigma2=0.05)
        L = affinity_to_normalized_laplacian(A)
        L_sparse = sparse.csr_matrix(L)
        
        # Run community detection with pre-computed affinity matrix
        detector = CommunityDetector(max_communities=10, random_state=42, 
                                    affinity='precomputed')
        labels = detector.fit_predict(L_sparse)
        
        # Should detect 3 communities
        assert detector.n_communities_ == 3, \
            f"Expected 3 concentric circles, detected {detector.n_communities_}"
        
        # Check purity for each true circle
        for true_label in range(3):
            mask = true_labels == true_label
            pred_in_cluster = labels[mask]
            
            # Most common predicted label should dominate (>90% purity for this clean case)
            most_common = np.bincount(pred_in_cluster).argmax()
            purity = (pred_in_cluster == most_common).sum() / len(pred_in_cluster)
            
            assert purity > 0.9, \
                f"Concentric circle {true_label} (r={true_label+1}) purity {purity:.2%} < 90%"


class TestHelperFunctions:
    """Test helper functions for community detection."""
    
    def test_gaussian_affinity(self):
        """Test Gaussian affinity computation."""
        X = np.array([[0, 0], [1, 0], [10, 10]])
        A = compute_gaussian_affinity(X, sigma2=1.0)
        
        # Check symmetry
        assert np.allclose(A, A.T)
        
        # Check diagonal is 1
        assert np.allclose(np.diag(A), 1.0)
        
        # Check nearby points have high affinity
        assert A[0, 1] > 0.3
        
        # Check distant points have low affinity
        assert A[0, 2] < 0.01
    
    def test_normalized_laplacian(self):
        """Test normalized Laplacian computation."""
        A = np.array([
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.5],
            [0.0, 0.5, 1.0]
        ])
        
        L = affinity_to_normalized_laplacian(A)
        
        # Check symmetry
        assert np.allclose(L, L.T)
        
        # Check eigenvalue properties (largest eigenvalue should be ~1)
        eigvals = np.linalg.eigvalsh(L)
        assert eigvals[-1] <= 1.0 + 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
