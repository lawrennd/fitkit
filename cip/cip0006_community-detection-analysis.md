---
author: "Neil Lawrence"
created: "2026-02-06"
id: "0006"
last_updated: "2026-02-06"
status: "Proposed"
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

- [x] Proposed
- [ ] Accepted
- [ ] In Progress
- [ ] Implemented
- [ ] Closed
- [ ] Rejected
- [ ] Deferred

## Summary

Add community detection capabilities to the `fitkit` library core:
1. **Library integration**: New module `fitkit.community` with eigenvector-based community detection
2. **Sklearn-style interface**: `CommunityDetector` class following fitkit conventions
3. **Within-community analysis**: Utilities to compute ECI/Fitness separately per community
4. **Notebook demonstration**: Update `spectral_entropic_comparison.ipynb` to demonstrate library features
5. **Data-driven diagnostics**: Replace rigid thresholds with qualitative patterns

**Key architectural decision**: Community detection is core functionality, not just a notebook helper. It belongs in the library proper, with tests and documentation.

## Motivation

### Problems to solve

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

### Why proper library integration matters

A prototype implementation was created in `examples/community_analysis_helpers.py`, but this is architecturally wrong:

**Problems with notebook-only helper**:
- Not reusable across projects
- Not tested
- Not discoverable to library users
- Doesn't follow fitkit's sklearn-style conventions
- Creates duplicate functionality risk

**Correct approach**:
- Community detection is general spectral analysis - belongs in library core
- Should follow sklearn-style interface like `ECI` and `FitnessComplexity`
- Should be tested and documented
- Notebook should **demonstrate** library features, not implement them

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

**Trade-offs**: Eigengap heuristic is simple, principled, and aligns with spectral theory foundations already in the paper. May miss hierarchical structure, but that's acceptable for initial implementation.

#### API Design Alternatives

**Option A: Sklearn-style class (RECOMMENDED)**:
```python
detector = CommunityDetector(method='spectral', n_communities='auto')
labels = detector.fit_predict(M)
print(detector.n_communities_, detector.eigenvalues_)
```

**Pros**: Consistent with `ECI` and `FitnessComplexity`, stateful, exposes diagnostics
**Cons**: More boilerplate for simple use

**Option B: Functional API**:
```python
labels, n_communities = detect_communities(M, method='spectral')
```

**Pros**: Simpler for one-off use
**Cons**: Inconsistent with fitkit patterns, harder to extend, diagnostics awkward

**Decision**: Use Option A (sklearn-style) for consistency with existing fitkit API conventions. This is the pattern established by CIP-0004.

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

**Proposed library structure**:

```
fitkit/
├── community/
│   ├── __init__.py
│   ├── detection.py          # CommunityDetector class
│   └── analysis.py            # within_community_analysis()
```

**Core components**:

1. **`CommunityDetector` class** (sklearn-style):
   ```python
   from fitkit.community import CommunityDetector
   
   detector = CommunityDetector(method='spectral', n_communities='auto')
   labels = detector.fit_predict(M)  # Returns community labels
   ```

2. **Utility functions**:
   ```python
   from fitkit.community import within_community_analysis
   
   stats = within_community_analysis(M, labels, metrics=['eci', 'fitness'])
   ```

**Notebook becomes demonstration**:
- Import from `fitkit.community`
- Show how to use the library features
- Visualize results
- Provide interpretation guidance

**Why this architecture**:
- Community detection is general spectral analysis - not notebook-specific
- Reusable across projects
- Testable and maintainable
- Discoverable via library API
- Follows fitkit's sklearn-style conventions

## Implementation Plan

1. **Create library module structure**
   - [ ] Create `fitkit/community/` directory
   - [ ] Create `__init__.py` with exports
   - [ ] Add to `fitkit/__init__.py` imports

2. **Implement `CommunityDetector` class** (`fitkit/community/detection.py`)
   - [ ] `__init__(method='spectral', n_communities='auto', max_communities=5)`
   - [ ] `fit(M)` - compute eigenvalues/vectors, detect communities
   - [ ] `fit_predict(M)` - fit and return labels
   - [ ] `labels_` attribute - community assignments after fitting
   - [ ] `n_communities_` attribute - number detected
   - [ ] `eigenvalues_` attribute - for diagnostics

3. **Implement analysis utilities** (`fitkit/community/analysis.py`)
   - [ ] `within_community_analysis(M, labels, metrics=['eci', 'fitness'])`
   - [ ] Returns per-community statistics (correlations, sizes, etc.)
   - [ ] Handles edge cases (small communities, sparse networks)

4. **Add tests** (`tests/test_community_detection.py`)
   - [ ] Test on nested network (should detect 1 community)
   - [ ] Test on modular network (should detect 2+ communities)
   - [ ] Test eigengap heuristic behavior
   - [ ] Test within-community analysis
   - [ ] Test edge cases (small networks, degenerate cases)

5. **Update notebook** (`examples/spectral_entropic_comparison.ipynb`)
   - [ ] Import from `fitkit.community`
   - [ ] Add community detection demonstration section
   - [ ] Show within-community analysis
   - [ ] Replace rigid diagnostic thresholds with qualitative patterns
   - [ ] Add visualization of community structure

6. **Documentation**
   - [ ] Docstrings for all public methods
   - [ ] Usage example in module docstring
   - [ ] Update README if appropriate

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

**Current state**: Prototype exists in `examples/community_analysis_helpers.py` but needs proper integration.

- [ ] Create `fitkit/community/` module structure
- [ ] Implement `CommunityDetector` sklearn-style class
- [ ] Implement `within_community_analysis()` utility
- [ ] Add comprehensive tests
- [ ] Update notebook to use library (not helpers)
- [ ] Remove prototype helper file
- [ ] Add documentation

## References

**Theoretical foundations**:
- von Luxburg, U. (2007). "A tutorial on spectral clustering." *Statistics and Computing* 17(4), 395-416.
- Newman, M. E. (2006). "Modularity and community structure in networks." *PNAS* 103(23), 8577-8582.

**Economic complexity literature**:
- Hidalgo & Hausmann (2009). "The building blocks of economic complexity." *PNAS* 106(26), 10570-10575.
- Tacchella et al. (2012). "A new metrics for countries' fitness and products' complexity." *Scientific Reports* 2, 723.
- Balland & Rigby (2016). "The geography of complex knowledge." *Economic Geography* 93(1), 1-23.

**Code status**:
- `examples/community_analysis_helpers.py` - Prototype (to be replaced by library module)
- `examples/spectral_entropic_comparison.ipynb` - Has preliminary integration (needs update)
- Target: `fitkit/community/` module (to be created)

## Future Enhancements

Potential extensions (not in scope for this CIP):
1. Robustness analysis: Bootstrap resampling to assess community detection stability
2. Hierarchical communities: Recursive application to detect nested structure
3. Product-side communities: Analogous analysis for product space (via ψ₂ᴾ, ψ₃ᴾ, ...)
4. Directed graphs: Extend to directed trade networks
5. Temporal evolution: Track how communities change over time
