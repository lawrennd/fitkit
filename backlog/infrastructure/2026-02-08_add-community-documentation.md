---
id: "2026-02-08_add-community-documentation"
title: "Add documentation for community detection module"
status: "Proposed"
priority: "Medium"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "documentation"
related_cips: ["0006"]
owner: "Neil Lawrence"
dependencies: 
- "2026-02-08_implement-community-detector"
- "2026-02-08_implement-validation-utilities"
- "2026-02-08_implement-analysis-utilities"
tags:
- backlog
- community-detection
- documentation
---

# Task: Add Documentation for Community Detection Module

> **Note**: Backlog tasks are DOING the work defined in CIPs (HOW).  
> Use `related_cips` to link to CIPs. Don't link directly to requirements (bottom-up pattern).

## Description

Create comprehensive documentation for the new `fitkit.community` module, including API reference, user guide, and theoretical background.

## Acceptance Criteria

- [ ] **API Reference** (`docs/api/community.rst`):
  - `CommunityDetector` class documentation
  - Validation functions documentation
  - Analysis functions documentation
  - All parameters, attributes, methods documented
  - Examples for each function

- [ ] **User Guide** (`docs/user_guide/community_detection.rst`):
  - Introduction to community detection in economic networks
  - When to use community detection
  - Basic usage tutorial
  - Interpreting results
  - Common pitfalls and best practices

- [ ] **Theory Section** (`docs/theory/community_detection.rst`):
  - Overview of iterative eigenvector algorithm
  - Explanation of elongated k-means
  - Diffusion maps interpretation
  - Validation methods theory (permutation tests, Cheeger, effective rank)
  - Connections to economic-fitness.tex insights

- [ ] **Algorithm Details** (`docs/algorithms/iterative_clustering.rst`):
  - Step-by-step algorithm description
  - Pseudo-code
  - Convergence criteria
  - Parameter tuning guidelines
  - References to Sanguinetti, Lawrence & Laidler (2005)

- [ ] **Docstrings**:
  - All classes have comprehensive docstrings
  - All functions have docstrings with Args/Returns/Examples
  - Docstrings follow NumPy style
  - Code examples in docstrings are tested

## Implementation Notes

**API reference template**:
```rst
CommunityDetector
-----------------

.. autoclass:: fitkit.community.CommunityDetector
   :members:
   :inherited-members:
   :show-inheritance:

   **Examples**

   Basic usage::

       from fitkit.community import CommunityDetector
       
       detector = CommunityDetector(n_communities='auto')
       labels = detector.fit_predict(M)
       print(f"Found {detector.n_communities_} communities")
```

**User guide structure**:
1. Introduction
2. Installation and imports
3. Quick start example
4. Understanding the output
5. Advanced usage (validation, analysis)
6. Real-world example
7. Troubleshooting

**Theory section highlights**:
- Explain why eigenvectors capture community structure (diffusion maps)
- Explain why radial elongation occurs (insufficient dimensionality)
- Explain how iterative algorithm finds intrinsic dimensionality
- Link to economic-fitness.tex theoretical framework

**Cross-references**:
- Link to ECI documentation (complementary methods)
- Link to spectral_entropic_comparison.ipynb example
- Link to economic-fitness.tex paper

## Related

- CIP: 0006
- Documentation structure follows existing fitkit patterns
- References: Sanguinetti, Lawrence & Laidler (2005), economic-fitness.tex (2026)

## Progress Updates

### 2026-02-08

Task created from CIP-0006 acceptance.
