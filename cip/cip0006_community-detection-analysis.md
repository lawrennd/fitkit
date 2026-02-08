---
author: "Neil Lawrence"
created: "2026-02-06"
id: "0006"
last_updated: "2026-02-06"
status: "In Progress"
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
- [x] Accepted
- [x] In Progress
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

**Key theoretical advance**: The 2026 economic-fitness paper transforms the 2005 iterative algorithm from a heuristic into a principled geometric method by connecting it to diffusion maps, separation timescales, Cheeger bounds, and intrinsic dimensionality. This provides rigorous justification for the origin-detector approach and explains the complementarity revealed by within-community analysis (Morphology B).

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

**A. Pure eigengap heuristic** (original CIP draft):
- Find largest gap in eigenvalue spectrum, use k = argmax(Δᵢ) + 1
- **Problem**: Arbitrary threshold, no validation, no statistical justification
- **Verdict**: Rejected - too naive

**B. Effective rank elbow detection**:
- Compute R(t) = exp(-Σᵢ pᵢ log pᵢ) curve
- Find elbow at t*, use that as dimensionality indicator
- **Advantage**: Continuous, principled dimensionality diagnostic
- **Problem**: Elbow detection itself requires thresholds or subjective judgment
- **Verdict**: Use as **complementary diagnostic**, not primary method

**C. Modularity maximization** (Louvain, etc.):
- Different objective function (within vs between community edge density)
- Requires resolution parameter tuning
- **Verdict**: Different purpose - use for modularity-specific questions

**D. Infomap**:
- Flow-based communities using random walk compression
- **Verdict**: Overkill, assumes directed/weighted, less natural for bipartite capability networks

**E. Modern manifold-aware k-means** (2020+):
- Manifold adaptive multiple kernel k-means explicitly incorporates local manifold structure
- Uses multiple kernels + manifold regularization
- **Advantage**: More sophisticated handling of nonlinear manifold geometry
- **Problem**: Significantly more complex, requires kernel selection, unclear benefit for bipartite networks
- **Verdict**: Overkill for initial implementation - the 2005 approach already operates in manifold (eigenvector) space

**F. Kernel k-means equivalence** (theoretical insight):
- Spectral clustering = kernel k-means in eigenvector embedding (Schölkopf, Wainwright et al. 2015)
- The 2005 approach already exploits this: k-means in eigenvector space is manifold-aware
- Elongated metric adapts to the radial structure of insufficient-dimensional projections
- **Verdict**: Confirms the 2005 approach is already manifold-aware; modern variants add complexity without clear benefit for our use case

**Decision**: Use iterative algorithm (Sanguinetti 2005) as **primary method** with effective rank R(t) and validation tests (permutation, Cheeger) as **diagnostics**. 

**Rationale**: 
- The 2005 approach is already manifold-aware (operates in eigenvector/diffusion space)
- Empirically validated, computationally efficient
- Now theoretically grounded via 2026 economic-fitness diffusion map interpretation
- Modern manifold-adaptive variants add complexity without demonstrated benefit for bipartite networks
- Can revisit if empirical results show inadequacy

#### Elongated K-Means Details

Key modification from standard k-means: distance metric that downweights radial direction, penalizes tangential direction.

For center **c**ᵢ not near origin, distance from point **x** is:

d²(**x**, **c**ᵢ) = (**x** - **c**ᵢ)ᵀ **M** (**x** - **c**ᵢ)

where **M** = (1/λ)(I - **c**ᵢ**c**ᵢᵀ/||**c**ᵢ||²) + λ(**c**ᵢ**c**ᵢᵀ/||**c**ᵢ||²)

- λ is elongation parameter (default 0.2)
- Small λ → strong elongation along radial direction
- For center at origin, use Euclidean distance

**Why this works**: Clusters in eigenvector space with insufficient dimensions appear as radial elongations. This metric makes them separable, and a center at the origin will only capture points if there's an unaccounted radial cluster.

### Theoretical Foundation: Diffusion Maps and Intrinsic Dimensionality

The 2026 economic-fitness paper provides deep theoretical justification for why the iterative algorithm works:

#### 1. Higher Eigenvectors as Geometric Modes

From diffusion map theory, eigenvectors ψ₂, ψ₃, ψ₄, ... represent progressively finer-grained geometric modes of the network. When k-dimensional structure is projected into q < k dimensions:

- **Missing modes manifest as radial structure**: The unaccounted eigenvectors create elongations along radial directions
- **Origin-detector principle**: Points from an unaccounted cluster align along a radial direction not yet captured by the current q eigenvectors
- **Iterative reveal**: Adding eigenvector q+1 "unfolds" that radial elongation into a new geometric mode

This transforms the 2005 heuristic into a **principled geometric method**: you're progressively revealing the intrinsic dimensionality by testing when diffusion has exposed all persistent modes.

#### 2. Spectral Gap as Intrinsic Dimensionality

The gap λ₃ᴸ - λ₂ᴸ is not an arbitrary threshold but a **rigorous dimensionality diagnostic**:

- **Large gap (>2-3)**: Capability space is effectively 1D, single nested hierarchy
- **Small gap (<1.5)**: Multi-scale structure, need multiple eigenvectors for full embedding
- **Physical interpretation**: Gap measures separation between the dominant mode and finer structure

**Separation timescale**: t* ≈ 1/(λ₃ᴸ - λ₂ᴸ)

- Low-conductance communities: persist and separate slowly (large t*, small gap)
- Nested hierarchies: collapse quickly to ψ₂ mode (small t*, large gap)

The algorithm terminates when adding dimensions wouldn't reveal structure that persists under diffusion.

#### 3. Cheeger Inequality: Rigorous Bounds

Cheeger's inequality: **λ₂ᴸ ≥ Φ²/2**

This bounds the **mixing time**: τ_mix ≈ 1/λ₂ᴸ ≤ 2/Φ²

**Implications**:
- Low conductance Φ (bottlenecks) → long mixing times → communities persist under diffusion
- Connection between spectral gaps and network cuts is **quantitative, not heuristic**
- Validates detected communities: if Φ ≈ 0, the cut is real; if Φ is moderate, the small eigenvalue reflects 1D geometry, not communities

#### 4. Morphology B: The Complementarity Pattern

**Critical insight from economic-fitness paper**: What the algorithm detects is **complementary perspectives**, not disagreement:

For modular networks (Morphology B):
- **Global discordance**: Low r_global between ECI & Fitness (< 0.5)
- **Within-community concordance**: High r_comm (> 0.8) when analyzed separately per community
- **ψ₂ acts as block indicator**: Positive vs negative sign labels communities
- **Fitness measures within-block capability**: Harmonic aggregation operates within communities

**The origin-detector finds this structure**: When undercounted, different communities project onto radial directions. The origin center captures points from the unaccounted community, triggering the addition of another eigenvector that separates the blocks.

#### 5. Effective Rank: Alternative to Fixed Thresholds

Instead of arbitrary eigengap thresholds, use **effective rank** as diffusion time varies:

**R(t) = exp(-Σᵢ pᵢ log pᵢ)** where pᵢ ∝ e^(-λᵢᴸ t)

Plot R(t) vs diffusion time t:
- **Sharp elbow at t*** → intrinsic dimensionality revealed (number of persistent modes)
- Nested networks: steep elbow at small t (collapse to 1D quickly)
- Multi-scale networks: gradual decay or multiple elbows at different timescales

This provides a **data-driven alternative** to fixed thresholds: the elbow location reveals how many eigenvectors are needed.

**Implementation consideration**: Computing R(t) curve could replace or supplement the iterative origin-detector test. Both approaches test intrinsic dimensionality, but R(t) is continuous while the algorithm is discrete.

#### 6. The 2005 Algorithm is Already Manifold-Aware

**Important clarification**: The Sanguinetti-Lawrence-Laidler 2005 algorithm operates in **eigenvector space**, which is itself a **diffusion map embedding** of the manifold structure.

**Key insight from spectral clustering theory** (Schölkopf, Wainwright et al. 2015):
- Spectral clustering = kernel k-means in the eigenvector embedding
- The eigenvectors ψ₂, ψ₃, ... provide a **nonlinear manifold embedding** of the original data
- K-means in this space is **implicitly manifold-aware** - it's clustering on the diffusion geometry, not raw features

**What the 2005 algorithm adds**:
- **Elongated distance metric**: Adapts to the specific radial structure that appears when k-dimensional manifold structure is projected into q < k eigenvector dimensions
- **Origin-detector**: Tests whether the current embedding dimension q has captured all persistent modes
- **Iterative dimension increase**: Progressively reveals manifold dimensionality by adding eigenvectors

**Modern manifold-adaptive k-means** (2020+): Incorporate local manifold structure explicitly via manifold regularization and multiple kernels. This is **more sophisticated** but:
- The 2005 approach already operates on manifold (via eigenvector embedding)
- Added sophistication may not help for bipartite networks where eigenvector structure is well-characterized
- Computational cost increases significantly

**Conclusion**: The 2005 algorithm is **not a naive k-means** - it's k-means on a diffusion map embedding with an adaptive metric for insufficient dimensions. Modern methods add bells and whistles, but the core principle (clustering in manifold coordinates) is already there.

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

# Dimensionality diagnostic via effective rank
from fitkit.community.validation import compute_effective_rank
t_range = np.logspace(-2, 2, 100)
R_t = compute_effective_rank(detector.eigenvalues_, t_range)

# Plot R(t) curve to visualize dimensionality
plt.plot(t_range, R_t)
plt.xlabel('Diffusion time t')
plt.ylabel('Effective rank R(t)')
plt.title('Intrinsic Dimensionality via Effective Rank')
# Look for elbow - sharp drop indicates dominant modes
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

**Solution 4: Effective rank for intrinsic dimensionality**

Compute effective rank as function of diffusion time:

R(t) = exp(-Σᵢ pᵢ log pᵢ) where pᵢ ∝ e^(-λᵢᴸ t)

- Plot R(t) vs t
- Sharp elbow at t* indicates intrinsic dimensionality k
- Use k eigenvectors for embedding
- More principled than arbitrary gap thresholds

**Relationship to iterative algorithm**: 
- Effective rank is **continuous diagnostic** (smooth curve)
- Iterative algorithm is **discrete test** (active probing with origin detector)
- Both test same concept: intrinsic dimensionality
- R(t) could guide max_communities parameter or validate iterative results

**Implementation priority**: 
1. Core: Iterative algorithm (Solution from 2005 paper)
2. Validation: Permutation tests (Solution 1) + Cheeger bounds (Solution 2)
3. Diagnostic: Effective rank (Solution 4) as complementary dimensionality check
4. Refinement: Configuration model (Solution 3) for bipartite-specific null

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
   - [ ] `compute_effective_rank(eigenvalues, t_range)` - R(t) curve for dimensionality
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

### Unit Tests (`tests/test_community_detection.py`)

**Core functionality**:
- Iterative algorithm converges (origin detector terminates correctly)
- Elongated k-means behaves properly (radial vs Euclidean metrics)
- Edge cases: small networks, degenerate matrices, no structure

**Expected behaviors on synthetic networks**:
- **Nested network**: Detects 1 community (origin remains empty from q=2)
- **Modular network**: Detects 2 communities, origin captures points at q=2, empty at q=3
- **Multi-scale network**: May detect multiple communities (q>2) based on eigengap persistence

**Statistical validation tests**:
- Permutation test: Known structure has p-value < 0.05, random network has p-value > 0.5
- Cheeger bounds: Detected communities satisfy λ₂ᴸ ≥ Φ²/2
- Effective rank: R(t) curve shows elbow consistent with detected k

**Within-community analysis**:
- Modular network: r_global < 0.5 but r_within > 0.8 (Morphology B pattern)
- Nested network: r_global ≈ r_within (no community structure)

### Integration Tests (notebook execution)

Manual testing via `spectral_entropic_comparison.ipynb`:
- All synthetic morphologies run without errors
- Visualizations render correctly
- Validation metrics are interpretable
- Effective rank plots show expected elbows

**Key validation**: For modular network, within-community correlations should be **substantially higher** than global correlation, validating the complementarity hypothesis.

## Related Requirements

None formally defined. This addresses user feedback about:
- Using eigenvectors for community separation
- Making diagnostic assumptions less rigid and more realistic

## Implementation Status

**Current state**: Prototype exists in `examples/community_analysis_helpers.py` with:
- Basic eigengap heuristic (naive, no validation)
- Simple k-means clustering (not elongated k-means)
- Within-community analysis logic
- Integrated into notebook cells 23-25

**This prototype is architecturally wrong** - needs proper library integration and theoretical rigor.

**Remaining work**:
- [x] Create `fitkit/community/` module structure
- [x] Implement `CommunityDetector` with iterative algorithm (2005 paper)
- [x] Implement elongated k-means (Mahalanobis radial distance)
- [x] Implement origin-detector termination criterion
- [ ] Implement validation utilities (permutation, Cheeger, effective rank)
- [ ] Add comprehensive tests (including validation tests)
- [ ] Update notebook to use library (replace helper imports)
- [ ] Remove prototype helper file after migration
- [x] Add docstrings and module documentation (for CommunityDetector)

## References

**Core algorithm**:
- Sanguinetti, G., Laidler, J., & Lawrence, N. D. (2005). "Automatic determination of the number of clusters using spectral algorithms." *Proceedings of the 14th International Conference on Digital Signal Processing*, 717-721. [PDF](https://www.math.ucdavis.edu/~saito/data/clustering/clusterNumber.pdf) | [Code](https://github.com/lawrennd/spectral)

**Theoretical foundation** (transforms 2005 heuristic into principled method):
- Lawrence, N. D. (2026). "Economic fitness and complexity: Spectral and entropic perspectives on bipartite networks." [Draft in progress: `economic-fitness/economic-fitness.tex`]
  - §3.5-3.7: Diffusion map interpretation, ECI as geometric embedding, intrinsic dimensionality
  - §3.8: Morphology classification, Cheeger diagnostics, spectral-entropic comparison plots
  - Lines 489-518: Why higher eigenvectors matter, separation timescales, effective rank
  - Lines 633-645: Morphology B complementarity - low global r but high within-community r
  - **Key synthesis**: The 2005 algorithm's origin-detector iteratively reveals intrinsic dimensionality by testing when diffusion has exposed all persistent geometric modes. Elongated k-means exploits the radial projection structure that appears when multi-scale manifold geometry is embedded in insufficient dimensions.

**Foundational spectral theory**:
- von Luxburg, U. (2007). "A tutorial on spectral clustering." *Statistics and Computing* 17(4), 395-416.
- Coifman, R. R. & Lafon, S. (2006). "Diffusion maps." *Applied and Computational Harmonic Analysis* 21(1), 5-30.
- Belkin, M. & Niyogi, P. (2003). "Laplacian eigenmaps for dimensionality reduction and data representation." *Neural Computation* 15(6), 1373-1396.
- Ng, A. Y., Jordan, M. I., & Weiss, Y. (2001). "On spectral clustering: Analysis and an algorithm." *NIPS* 14.

**Manifold-aware clustering (modern perspective)**:
- Schölkopf, B., Wainwright, M. J., & Yu, B. (2015). "The geometry of kernelized spectral clustering." *The Annals of Statistics* 43(2), 819-846. [Establishes spectral clustering = kernel k-means equivalence]
- Liu, W., et al. (2020). "Manifold adaptive multiple kernel k-means for clustering." *IEEE Access* 8, 184389-184401. [Modern manifold-adaptive variant]

**Graph cut theory**:
- Cheeger, J. (1970). "A lower bound for the smallest eigenvalue of the Laplacian." *Problems in Analysis*, 195-199.
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

Potential extensions beyond initial implementation (not in scope for this CIP):

**1. Robustness and stability**:
- Bootstrap resampling to assess community detection stability
- Consensus clustering across multiple parameter settings
- Sensitivity analysis for λ_elongation parameter

**2. Hierarchical structure**:
- Recursive application within detected communities
- Dendrogram construction from nested eigenvector splits
- Multi-resolution analysis at different diffusion times

**3. Product-side communities**:
- Analogous detection using product eigenvectors (ψ₂ᴾ, ψ₃ᴾ, ...)
- Product clusters as technological domains
- Cross-tabulation: country communities × product communities

**4. Temporal evolution**:
- Track community membership changes over time
- Detect community mergers, splits, emergence, dissolution
- Relate to economic shocks and structural transformation

**5. Alternative spectral methods**:
- Heat kernel diffusion: K_t = exp(-tL) for continuous-time perspective
- Multi-scale analysis: vary diffusion time t to reveal structure at different scales
- Directed networks: left/right eigenvectors for asymmetric trade flows

**6. Modern manifold-adaptive methods**:
- Investigate manifold adaptive multiple kernel k-means (Liu et al. 2020)
- Test if explicit manifold regularization improves detection on bipartite networks
- Compare against iterative 2005 approach on challenging real datasets
- **Question**: Does added sophistication help? Or is 2005 approach sufficient given bipartite structure?
- Empirical evaluation needed before adding complexity

**7. Causal structure**:
- Use community detection to identify natural experiments
- Within-community comparisons as matched controls
- Eigenvector discontinuities as potential policy boundaries
