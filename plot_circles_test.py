#!/usr/bin/env python
"""
Generate visualization of three circles test result.
Based on MATLAB demoCircles.m
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import sparse
import sys
sys.path.insert(0, '.')

from fitkit.community.detection import CommunityDetector


def compute_gaussian_affinity(X, sigma2):
    """Compute Gaussian affinity matrix."""
    n = len(X)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = np.exp(-np.linalg.norm(X[i] - X[j])**2 / sigma2)
    return A


def affinity_to_normalized_laplacian(A):
    """Convert affinity to normalized Laplacian."""
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    L = D_inv_sqrt @ A @ D_inv_sqrt
    return L


# Generate three circles
np.random.seed(42)
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

# Combine and add noise
X = np.vstack([circle1, circle2, circle3])
X += np.random.randn(*X.shape) * 0.05

true_labels = np.array([0]*n_per_circle + [1]*n_per_circle + [2]*n_per_circle)

print("Generated 3 circles with 50 points each")
print(f"Data shape: {X.shape}")

# Compute affinity and Laplacian
sigma2 = 0.05
print(f"\nComputing Gaussian affinity (sigma2={sigma2})...")
A = compute_gaussian_affinity(X, sigma2)
L = affinity_to_normalized_laplacian(A)
L_sparse = sparse.csr_matrix(L)

# Run community detection
print("Running community detection...")
detector = CommunityDetector(max_communities=10, random_state=42)
labels = detector.fit_predict(L_sparse)

print(f"\nDetected {detector.n_communities_} communities")
print(f"Iterations: {detector.n_iterations_}")

# Compute purity for each circle
print("\nPurity by true circle:")
for true_label in range(3):
    mask = true_labels == true_label
    pred_in_cluster = labels[mask]
    most_common = np.bincount(pred_in_cluster).argmax()
    purity = (pred_in_cluster == most_common).sum() / len(pred_in_cluster)
    print(f"  Circle {true_label}: {purity:.1%} (assigned to cluster {most_common})")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: True labels
ax = axes[0]
for i in range(3):
    mask = true_labels == i
    ax.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.7, label=f'Circle {i}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('True Labels (3 Circles)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Plot 2: Predicted labels
ax = axes[1]
unique_pred = np.unique(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pred)))
for i, pred_label in enumerate(unique_pred):
    mask = labels == pred_label
    ax.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.7, 
               label=f'Cluster {pred_label} (n={mask.sum()})', color=colors[i])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'Predicted Labels ({detector.n_communities_} Communities)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('circles_test_result.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved to circles_test_result.png")

plt.show()
