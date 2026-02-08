---
author: "Neil Lawrence"
created: "2026-02-06"
id: "0006"
last_updated: "2026-02-06"
status: "Implemented"
compressed: false
related_requirements: []
related_cips: []
tags:
- cip
- community-detection
- spectral-analysis
- eigenvectors
title: "Community Detection and Within-Community Analysis for Spectral-Entropic Comparison"
---

# CIP-0006: Community Detection and Within-Community Analysis for Spectral-Entropic Comparison

## Status

- [ ] Proposed
- [ ] Accepted
- [ ] In Progress
- [x] Implemented
- [ ] Closed
- [ ] Rejected
- [ ] Deferred

## Summary

Enhance the `spectral_entropic_comparison.ipynb` notebook with:
1. Eigenvector-based community detection using the eigengap heuristic
2. Within-community analysis to compare global vs local ECI-Fitness correlations
3. Data-driven qualitative diagnostics replacing rigid categorical thresholds

This addresses the observation that sub-communities can be separated using higher eigenvectors and analyzed independently, revealing complementary perspectives (ECI for community boundaries, Fitness for within-community capability gradients).

## Motivation

The original notebook had two limitations:

**1. Rigid diagnostic thresholds created false confidence**
- Hard cutoffs like "if r > 0.85 and gap > 2 then Morphology A" don't match real networks
- Real economic networks rarely fit clean categories
- Arbitrary thresholds discouraged visual inspection and domain knowledge

**2. No community detection despite eigenvector availability**
- User insight: "When we have sub-communities, can't we use the eigenvectors to separate and study each sub-community separately?"
- The notebook computed multiple eigenvectors but only used the first (Fiedler vector)
- Morphology B (modular networks) show low global correlation but potentially high within-community correlation
- This complementarity wasn't being demonstrated

## Detailed Description

### Design Decisions

#### 1. Community Detection Approach

**Chosen: Spectral clustering with eigengap heuristic**

Uses k-means clustering on the eigenvector embedding (ψ₂, ψ₃, ..., ψₖ) where k is determined by the eigengap heuristic.

**Eigengap Heuristic Algorithm**:
1. Compute Laplacian eigenvalues: λᴸᵢ = 1 - λᵢᵀ (where λᵢᵀ are transition matrix eigenvalues)
2. Compute gaps: Δᵢ = λᴸᵢ₊₁ - λᴸᵢ for i ≥ 1 (skip trivial eigenvalue)
3. Find largest gap: k* = argmax Δᵢ
4. Number of communities: k = k* + 1
5. Only use if gap is significant (>10% of λᴸ₁) to avoid over-fitting

**Why this works**: In networks with k communities, the first k Laplacian eigenvalues are small (near 0), with a large gap before λᴸₖ₊₁. This is the spectral graph theory foundation.

**Alternatives considered**:
- Modularity maximization (Louvain): More complex, requires tuning resolution parameter
- Infomap: Requires directed/weighted edges, overkill for demonstration
- Manual k selection: Defeats purpose of automatic detection

**Trade-offs**: Eigengap heuristic is simple, principled, and aligns with spectral theory foundations already in the paper. May miss hierarchical structure, but that's acceptable for demonstration purposes.

#### 2. Within-Community Analysis

For each detected community:
1. Extract sub-matrix: keep only rows (countries) in that community
2. Filter products: keep only products exported by community members  
3. Compute ECI and Fitness on sub-network using same sklearn-style estimators
4. Calculate correlation within community

**Key insight**: This properly accounts for community-specific product spaces and avoids artifacts from cross-community connections.

#### 3. Data-Driven Diagnostics

**Old approach** (rejected):
```python
if pearson_countries > 0.85 and gap_ratio_c > 2:
    print("Morphology A: Single nested hierarchy")
elif conductance < 0.15 and gap_ratio_c > 2:
    print("Morphology B: Low-conductance communities")
# etc.
```

**New approach** (implemented):
```python
# Report observed values
print(f"Correlation: {pearson_countries:.3f}")
print(f"Spectral gap: {gap_ratio_c:.3f}")
print(f"Conductance: {conductance:.3f}")

# Provide qualitative interpretation
if pearson_countries > 0.85:
    print("→ Very high: likely tight monotone trend")
# etc.
```

**Rationale**: 
- Real networks are messy and don't fit clean categories
- Reporting raw values encourages critical thinking
- Qualitative bands (very high, moderate, low) are more honest than precise cutoffs
- Emphasizes visual inspection over algorithmic classification

### Architecture

**New module**: `examples/community_analysis_helpers.py`

Contains two main functions:
- `detect_communities_from_eigenvectors(M, n_communities='auto', max_communities=5)`
- `analyze_within_communities(M, community_labels, ECI, FitnessComplexity)`

**Integration point**: New notebook cells 23-25
- Cell 23: Markdown explaining community detection approach
- Cell 24: Function `analyze_network_with_communities()` 
- Cell 25: Demo call on modular network

**Dependencies added**:
- `sklearn.cluster.KMeans` - for eigenvector clustering
- `scipy.sparse.linalg.eigs` - already used, now also in community detection

## Implementation Plan

**Already completed**:

1. **Create community detection helper module**
   - [x] Implement `detect_communities_from_eigenvectors()` with eigengap heuristic
   - [x] Implement `analyze_within_communities()` with per-community ECI/Fitness
   - [x] Add proper error handling for edge cases (small networks, sparse communities)

2. **Enhance notebook**
   - [x] Add imports for community analysis
   - [x] Add markdown cell explaining approach (cell 23)
   - [x] Add `analyze_network_with_communities()` function (cell 24)
   - [x] Add demo on modular network (cell 25)
   - [x] Replace rigid thresholds with qualitative patterns

3. **Visualization**
   - [x] Communities colored in ECI-Fitness space
   - [x] Eigenvalue spectrum with gaps marked
   - [x] Display global vs within-community correlations

## Backward Compatibility

Fully backward compatible:
- No changes to existing function signatures
- New functions are optional (only called if user runs new cells)
- Existing notebook cells still work identically
- No changes to `fitkit` library itself

## Testing Strategy

Manual testing via notebook execution:
- Nested network: Should detect 1 community (no meaningful splits)
- Modular network: Should detect 2 communities, show high within-community correlations
- Core-periphery: May detect 2 communities (core vs periphery)
- Multi-scale: May detect multiple communities based on eigengaps

**Validation criteria**:
- Community detection doesn't crash on edge cases
- Within-community analysis shows interpretable results
- Visualizations render correctly

## Related Requirements

None formally defined. This addresses user feedback about:
- Using eigenvectors for community separation
- Making diagnostic assumptions less rigid and more realistic

## Implementation Status

- [x] Create `community_analysis_helpers.py`
- [x] Implement eigengap heuristic
- [x] Implement within-community analysis
- [x] Add notebook cells 23-25
- [x] Update diagnostic interpretation sections
- [x] Add visualization code
- [x] Test on synthetic networks
- [ ] Validate on real economic data (future work)

## References

**Theoretical foundations**:
- von Luxburg, U. (2007). "A tutorial on spectral clustering." *Statistics and Computing* 17(4), 395-416.
- Newman, M. E. (2006). "Modularity and community structure in networks." *PNAS* 103(23), 8577-8582.

**Economic complexity literature**:
- Hidalgo & Hausmann (2009). "The building blocks of economic complexity." *PNAS* 106(26), 10570-10575.
- Tacchella et al. (2012). "A new metrics for countries' fitness and products' complexity." *Scientific Reports* 2, 723.
- Balland & Rigby (2016). "The geography of complex knowledge." *Economic Geography* 93(1), 1-23.

**Code files**:
- `examples/community_analysis_helpers.py` - Helper functions
- `examples/spectral_entropic_comparison.ipynb` - Enhanced notebook
- Previous enhancement doc: `examples/SPECTRAL_ENTROPIC_ENHANCEMENTS.md` (to be deleted after CIP compression)

## Future Enhancements

Potential extensions (not in scope for this CIP):
1. Robustness analysis: Bootstrap resampling to assess community detection stability
2. Hierarchical communities: Recursive application to detect nested structure
3. Product-side communities: Analogous analysis for product space (via ψ₂ᴾ, ψ₃ᴾ, ...)
4. Directed graphs: Extend to directed trade networks
5. Temporal evolution: Track how communities change over time
