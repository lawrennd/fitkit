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

**Chosen: Iterative eigenvector algorithm with origin-detector validation**

Based on Sanguinetti, Lawrence & Laidler (2005) "Automatic Determination of the Number of Clusters using Spectral Algorithms".

**Key insight from 2005 work**: When clusters are undercounted in eigenvector space, they appear as elongated radial structures. A test center at the origin will capture points if there's an unaccounted cluster.

**Algorithm**:
1. Start with q=2 eigenvectors
2. Initialize q centers in detected clusters + 1 center at origin
3. Run **elongated k-means** (Mahalanobis distance along radial directions)
4. If origin cluster captures points → unaccounted cluster exists → increment q
5. If origin cluster empty → found correct number of clusters → terminate

**Why this is better than naive eigengap thresholds**:
- Hard thresholds (e.g., "gap > 1.5") have no statistical justification
- Don't account for network size, density, or random fluctuations
- The iterative approach actively tests for additional structure
- Origin-detector provides empirical validation at each step

**Validation enhancements**:
1. **Permutation tests**: Test eigengaps against null distribution from degree-preserving randomization
2. **Cheeger bounds**: Validate detected communities via conductance φ(S) and Cheeger inequality
3. **Configuration model null**: For bipartite networks, test against random matrices with same (k_c, k_p) distributions

**Alternatives considered**:
- Pure eigengap heuristic (original CIP draft): Too naive, no validation
- Modularity maximization: Different objective, requires resolution tuning
- Infomap: Assumes flow-based communities, less natural for capability networks

**Trade-offs**: Iterative approach is more principled but requires more computation (multiple k-means runs). Worth it for statistical rigor.

#### Elongated K-Means Details

Key modification from standard k-means: distance metric that downweights radial direction, penalizes tangential direction.

For center **c**ᵢ not near origin, distance from point **x** is:

d²(**x**, **c**ᵢ) = (**x** - **c**ᵢ)ᵀ **M** (**x** - **c**ᵢ)

where **M** = (1/λ)(I - **c**ᵢ**c**ᵢᵀ/||**c**ᵢ||²) + λ(**c**ᵢ**c**ᵢᵀ/||**c**ᵢ||²)

- λ is elongation parameter (default 0.2)
- Small λ → strong elongation along radial direction
- For center at origin, use Euclidean distance

**Why this works**: Clusters in eigenvector space with insufficient dimensions appear as radial elongations. This metric makes them separable, and a center at the origin will only capture points if there's an unaccounted radial cluster.

#### API Design

**Sklearn-style class** (consistent with CIP-0004):
```python
from fitkit.community import CommunityDetector
from fitkit.community.validation import validate_communities

# Detect communities
detector = CommunityDetector(
    method='iterative',           # Sanguinetti et al. algorithm
    lambda_elongation=0.2,        # Radial elongation parameter
    n_communities='auto',         # Origin-detector termination
    max_communities=8
)
labels = detector.fit_predict(M)

# Access diagnostics
print(f"Found {detector.n_communities_} communities")
print(f"Eigenvalues: {detector.eigenvalues_[:5]}")
print(f"Iteration history: {detector.n_iterations_}")

# Validate detected structure
validation = validate_communities(M, labels, n_permutations=100)
print(f"Eigengap p-value: {validation['eigengap_pvalue']:.3f}")
print(f"Mean conductance: {validation['mean_conductance']:.3f}")
print(f"Significant structure: {validation['is_significant']}")
```

**Why sklearn-style**:
- Consistent with `ECI` and `FitnessComplexity` 
- Stateful - exposes iteration history, eigenvalues, validation scores
- Extensible - can add new methods, validation approaches
- Established pattern in fitkit (CIP-0004)

#### 2. Within-Community Analysis

For each detected community:
1. Extract sub-matrix: keep only rows (countries) in that community
2. Filter products: keep only products exported by community members  
3. Compute ECI and Fitness on sub-network using same sklearn-style estimators
4. Calculate correlation within community

**Key insight**: This properly accounts for community-specific product spaces and avoids artifacts from cross-community connections.

#### 3. Statistical Validation Methods

**Problem**: Need to distinguish real community structure from random fluctuations.

**Solution 1: Permutation tests for eigengaps**
```python
# Compute observed eigengap
observed_gap = eigenvalues[k] - eigenvalues[k+1]

# Generate null distribution via degree-preserving randomization
null_gaps = []
for _ in range(n_permutations):
    M_null = randomize_bipartite(M, preserve_degrees=True)
    eigs_null = compute_eigenvalues(M_null)
    null_gaps.append(eigs_null[k] - eigs_null[k+1])

# Test significance
p_value = (null_gaps >= observed_gap).mean()
```

**Solution 2: Cheeger bounds for community quality**

For each detected community S:
- Compute conductance: φ(S) = cut(S, S̄) / min(vol(S), vol(S̄))
- Cheeger inequality: λ₂ ≤ 2φ ≤ √(2λ₂)
- High conductance (φ > 0.5) indicates weak community structure
- Compare observed φ against null model

**Solution 3: Configuration model null for bipartite networks**

Generate random bipartite matrices with same degree sequences:
- Sample matrices preserving (k_c, k_p) distributions
- Compute eigenvalue spectra under null
- Test if observed eigengaps exceed null distribution
- More appropriate for bipartite economic networks than generic permutation

**Implementation priority**: Start with Solution 1 (permutation tests) and Solution 2 (Cheeger validation) as these are most straightforward. Add Solution 3 (configuration model) as refinement.

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
   - [ ] `__init__(method='iterative', n_communities='auto', max_communities=8, lambda_elongation=0.2)`
   - [ ] `fit(M)` - iterative eigenvector algorithm with origin detector
   - [ ] `fit_predict(M)` - fit and return labels
   - [ ] `_elongated_kmeans()` - Mahalanobis k-means with radial elongation
   - [ ] `labels_` attribute - community assignments after fitting
   - [ ] `n_communities_` attribute - number detected
   - [ ] `eigenvalues_` attribute - for diagnostics
   - [ ] `validation_scores_` - dictionary of validation metrics

3. **Implement analysis utilities** (`fitkit/community/analysis.py`)
   - [ ] `within_community_analysis(M, labels, metrics=['eci', 'fitness'])`
   - [ ] Returns per-community statistics (correlations, sizes, etc.)
   - [ ] Handles edge cases (small communities, sparse networks)

4. **Implement validation utilities** (`fitkit/community/validation.py`)
   - [ ] `permutation_test_eigengap(M, k, n_permutations=100, preserve_degrees=True)`
   - [ ] `compute_conductance(M, labels)` - Cheeger conductance for each community
   - [ ] `configuration_model_null(M, n_samples=100)` - bipartite-specific null
   - [ ] `validate_communities(M, labels)` - comprehensive validation report

5. **Add tests** (`tests/test_community_detection.py`)
   - [ ] Test iterative algorithm on nested network (should detect 1 community)
   - [ ] Test on modular network (should detect 2 communities, validate with high within-r)
   - [ ] Test origin-detector termination criterion
   - [ ] Test elongated k-means convergence
   - [ ] Test within-community analysis
   - [ ] Test permutation test (known structure should be significant)
   - [ ] Test conductance computation
   - [ ] Test edge cases (small networks, degenerate cases, no structure)

6. **Update notebook** (`examples/spectral_entropic_comparison.ipynb`)
   - [ ] Import from `fitkit.community`
   - [ ] Demonstrate iterative community detection on modular network
   - [ ] Show validation results (permutation tests, conductance)
   - [ ] Compare global vs within-community correlations
   - [ ] Replace rigid diagnostic thresholds with qualitative patterns
   - [ ] Visualize: communities in eigenvector space, eigenvalue spectrum with gaps
   - [ ] Show elongated k-means behavior on toy example

7. **Documentation**
   - [ ] Docstrings for all public methods (include math notation for elongated distance)
   - [ ] Usage example in module docstring
   - [ ] Document validation interpretation (p-values, conductance thresholds)
   - [ ] Reference Sanguinetti, Lawrence & Laidler (2005) paper

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

**Core algorithm**:
- Sanguinetti, G., Laidler, J., & Lawrence, N. D. (2005). "Automatic determination of the number of clusters using spectral algorithms." *Proceedings of the 14th International Conference on Digital Signal Processing*, 717-721. [PDF](https://www.math.ucdavis.edu/~saito/data/clustering/clusterNumber.pdf) | [Code](https://github.com/lawrennd/spectral)

**Theoretical foundations**:
- von Luxburg, U. (2007). "A tutorial on spectral clustering." *Statistics and Computing* 17(4), 395-416.
- Newman, M. E. (2006). "Modularity and community structure in networks." *PNAS* 103(23), 8577-8582.
- Cheeger, J. (1970). "A lower bound for the smallest eigenvalue of the Laplacian." *Problems in Analysis*, 195-199.

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
