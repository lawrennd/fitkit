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
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
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
    
    def test_circles_gaussian_affinity(self):
        """Test three circles example from MATLAB demoCircles.m.
        
        This tests the algorithm on synthetic data with Gaussian affinity
        matrix, matching the MATLAB reference implementation.
        """
        np.random.seed(42)
        
        # Generate three circles (matching MATLAB demoCircles.m)
        n_per_circle = 50
        
        # Circle 1: radius 1, center (0, 0)
        theta1 = np.linspace(0, 2*np.pi, n_per_circle)
        circle1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
        
        # Circle 2: radius 1, center (3, 0)
        theta2 = np.linspace(0, 2*np.pi, n_per_circle)
        circle2 = np.column_stack([3 + np.cos(theta2), np.sin(theta2)])
        
        # Circle 3: radius 1, center (1.5, 2.5)
        theta3 = np.linspace(0, 2*np.pi, n_per_circle)
        circle3 = np.column_stack([1.5 + np.cos(theta3), 2.5 + np.sin(theta3)])
        
        # Combine and add small noise
        X = np.vstack([circle1, circle2, circle3])
        X += np.random.randn(*X.shape) * 0.05
        
        true_labels = np.array([0]*n_per_circle + [1]*n_per_circle + [2]*n_per_circle)
        
        # Compute Gaussian affinity matrix (sigma2 = 0.05 as in MATLAB)
        A = compute_gaussian_affinity(X, sigma2=0.05)
        L = affinity_to_normalized_laplacian(A)
        L_sparse = sparse.csr_matrix(L)
        
        # Run community detection
        detector = CommunityDetector(max_communities=10, random_state=42)
        labels = detector.fit_predict(L_sparse)
        
        # Should detect 3 communities
        assert detector.n_communities_ == 3
        
        # Check purity for each true cluster
        for true_label in range(3):
            mask = true_labels == true_label
            pred_in_cluster = labels[mask]
            
            # Most common predicted label should dominate (>80% purity)
            most_common = np.bincount(pred_in_cluster).argmax()
            purity = (pred_in_cluster == most_common).sum() / len(pred_in_cluster)
            
            assert purity > 0.8, f"Circle {true_label} purity {purity:.2%} < 80%"


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
