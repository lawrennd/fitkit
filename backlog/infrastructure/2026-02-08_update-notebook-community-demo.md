---
id: "2026-02-08_update-notebook-community-demo"
title: "Update notebook to demonstrate library-based community detection"
status: "Proposed"
priority: "Medium"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "infrastructure"
related_cips: ["0006"]
owner: "Neil Lawrence"
dependencies: 
- "2026-02-08_implement-community-detector"
- "2026-02-08_implement-analysis-utilities"
tags:
- backlog
- community-detection
- documentation
- notebook
---

# Task: Update Notebook to Demonstrate Library-Based Community Detection

> **Note**: Backlog tasks are DOING the work defined in CIPs (HOW).  
> Use `related_cips` to link to CIPs. Don't link directly to requirements (bottom-up pattern).

## Description

Update `examples/spectral_entropic_comparison.ipynb` to use the new `fitkit.community` module instead of the prototype helper functions, demonstrating proper library usage.

## Acceptance Criteria

- [ ] Remove prototype `community_analysis_helpers.py` file
- [ ] Replace helper imports with `from fitkit.community import CommunityDetector, within_community_analysis`
- [ ] Update `analyze_network_with_communities()` function to use library classes
- [ ] Add new section demonstrating validation utilities:
  - Permutation test for eigengap significance
  - Cheeger bounds for community quality
  - Effective rank diagnostic for timescale analysis
- [ ] Add interpretation of validation results in markdown cells
- [ ] Update modular network demonstration to show:
  - Automatic community detection with iterative algorithm
  - Statistical validation of detected structure
  - Within-community analysis results
  - Global vs. local correlation comparison
- [ ] Add brief explanation of elongated k-means in markdown
- [ ] Ensure all cells run without errors

## Implementation Notes

**Before (prototype)**:
```python
from community_analysis_helpers import detect_communities_from_eigenvectors

labels = detect_communities_from_eigenvectors(M, n_communities='auto')
```

**After (library)**:
```python
from fitkit.community import CommunityDetector

detector = CommunityDetector(method='iterative', n_communities='auto')
labels = detector.fit_predict(M)

# Access diagnostics
print(f"Detected {detector.n_communities_} communities")
print(f"Converged in {detector.n_iterations_} iterations")
```

**New validation section**:
```python
from fitkit.community import validate_eigengap, compute_cheeger_bound, compute_effective_rank

# Test if structure is significant
gap, p_value = validate_eigengap(M, n_shuffles=100)
print(f"Eigengap: {gap:.3f}, p-value: {p_value:.3f}")

# Assess community quality
bounds = compute_cheeger_bound(M, labels)
for k, metrics in bounds.items():
    print(f"Community {k}: φ={metrics['conductance']:.3f}")

# Compute effective rank
R = compute_effective_rank(transition_matrix, t_max=10)
plt.plot(R)
plt.xlabel('Time t')
plt.ylabel('Effective Rank R(t)')
```

**Design note**: Keep notebook focused on demonstrating functionality, not implementation details. Link to library documentation for algorithms.

## Related

- CIP: 0006
- Notebook: `examples/spectral_entropic_comparison.ipynb`
- Prototype to remove: `examples/community_analysis_helpers.py`

## Progress Updates

### 2026-02-08

Task created from CIP-0006 acceptance.
