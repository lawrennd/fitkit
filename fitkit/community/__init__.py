"""Community detection and within-community analysis for bipartite networks.

This module provides tools for detecting communities in bipartite networks
using spectral methods and analyzing economic fitness measures within those
communities.

**Core Classes**:
- CommunityDetector: Iterative eigenvector-based community detection

**Analysis Functions**:
- within_community_analysis: Compute ECI/Fitness within each community
- compare_global_vs_local: Compare global vs. within-community correlations

**Validation Functions**:
- validate_eigengap: Permutation test for eigengap significance
- compute_cheeger_bound: Cheeger conductance bounds for community quality
- validate_bipartite_structure: Configuration model null test
- compute_effective_rank: Effective rank diagnostic for timescale analysis

The community detection algorithm is based on the iterative eigenvector method
from Sanguinetti, Lawrence & Laidler (2005), which automatically determines the
number of communities using an origin-detector approach with elongated k-means.

Theoretical foundations from diffusion maps (economic-fitness.tex, 2026) provide
insights into why the algorithm works: eigenvectors capture geometric modes,
spectral gaps measure separation timescales, and radial elongation occurs in
insufficient-dimensional projections.

Example:
    >>> from fitkit.community import CommunityDetector
    >>> from fitkit.algorithms import ECI, FitnessComplexity
    >>> 
    >>> # Detect communities
    >>> detector = CommunityDetector(n_communities='auto')
    >>> labels = detector.fit_predict(M)
    >>> print(f"Found {detector.n_communities_} communities")
    >>> 
    >>> # Analyze within communities
    >>> from fitkit.community import within_community_analysis
    >>> results = within_community_analysis(M, labels, ECI(), FitnessComplexity())

References:
    Sanguinetti, G., Laidler, J., & Lawrence, N. D. (2005).
    "Automatic determination of the number of clusters using spectral algorithms".
    IEEE MLSP.
    
    Lawrence, N.D. (2026). "Conditional Likelihood Interpretation of Economic Fitness".
"""

# Core classes
from fitkit.community.detection import CommunityDetector

# Placeholder imports for analysis functions (will be implemented in subsequent tasks)
# from fitkit.community.analysis import within_community_analysis, compare_global_vs_local
# from fitkit.community.validation import (
#     validate_eigengap,
#     compute_cheeger_bound,
#     validate_bipartite_structure,
#     compute_effective_rank,
# )

__all__ = [
    # Core classes
    "CommunityDetector",
    # Analysis functions (to be implemented)
    # "within_community_analysis",
    # "compare_global_vs_local",
    # Validation functions (to be implemented)
    # "validate_eigengap",
    # "compute_cheeger_bound",
    # "validate_bipartite_structure",
    # "compute_effective_rank",
]
