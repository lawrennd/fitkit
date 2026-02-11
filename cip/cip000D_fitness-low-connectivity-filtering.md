---
id: "000D"
title: "Fitness-Complexity Low Connectivity Filtering"
author: "Neil Lawrence"
created: "2026-02-11"
last_updated: "2026-02-11"
status: "Proposed"
related_requirements: []
related_cips: []
tags: ["fitness-complexity", "data-preprocessing", "numerical-stability"]
compressed: false
---

# CIP-000D: Fitness-Complexity Low Connectivity Filtering

## Status

- [ ] Proposed
- [ ] Accepted
- [ ] In Progress
- [ ] Implemented
- [ ] Closed
- [ ] Rejected
- [ ] Deferred

## Summary

The Fitness-Complexity algorithm exhibits numerical collapse when applied to bipartite graphs with low-connectivity nodes (products exported by only one country, or countries exporting very few products). This manifests as extreme concentration of fitness on a single country (e.g., Germany with F=234) while all other countries have near-zero fitness.

This CIP proposes adding preprocessing filters to remove low-connectivity nodes before running the Fitness-Complexity algorithm, following standard practice in the economic complexity literature.

## Motivation

### Problem

When analyzing Atlas of Economic Complexity trade data for 2020:
- Germany receives fitness of 234 (exactly the number of countries)
- All other major economies (USA, China, Japan, etc.) have fitness ≈ 0
- Only 3 out of 234 countries have fitness > 1e-10

Root cause: Germany exports 9 products that no other country exports (ubiquity = 1).

### Mechanism of Failure

The Fitness-Complexity fixed-point equations are:
```
F_c = sum_p M_cp * Q_p
Q_p = 1 / sum_c M_cp / F_c
```

When a product has ubiquity = 1 (exported by only one country):
1. That country gets high fitness from that product's complexity
2. The product gets high complexity (denominator has only one term)
3. This creates positive feedback: high Q → higher F → even higher Q
4. Meanwhile, other countries' products have lower complexity → lower fitness
5. The algorithm converges to a degenerate solution with one dominant country

### Mathematical Context

From `fitness.py` docstring:
> "The fixed point is unique up to the scale gauge **when the support graph is connected**."

Products with ubiquity = 1 violate the connectivity assumption, creating disconnected components in the bipartite graph.

### Observed Data

**2000 Data:**
- Products with ubiquity = 1: 8
- Countries with fitness > 1e-10: 3 (numerical collapse)
- Max fitness: 207.9
- Median fitness: 0.0

**2020 Data:**
- Products with ubiquity = 1: 10
- Countries with fitness > 1e-10: 3 (numerical collapse)
- Max fitness: 234.0
- Median fitness: 0.0

Both years exhibit the same pathology.

## Detailed Description

### Standard Practice in Literature

The original Tacchella et al. (2012) paper and subsequent implementations typically filter the trade matrix before running Fitness-Complexity:

1. **Remove low-ubiquity products**: Products exported by too few countries (e.g., ubiquity ≤ 2 or ≤ 5)
2. **Remove low-diversification countries**: Countries exporting too few products (e.g., diversification ≤ 5 or ≤ 10)
3. **Iteratively refilter**: After removing nodes, check again (removed nodes may affect connectivity)

### R Package Approach

The `economiccomplexity` R package filters out rows/columns with zero sums (see `test_r_fitness_comparison.R:43-52`):
```r
row_sums <- rowSums(M_matrix)
col_sums <- colSums(M_matrix)
valid_rows <- which(row_sums > 0)
valid_cols <- which(col_sums > 0)
M_filtered <- M_matrix[valid_rows, valid_cols, drop = FALSE]
```

However, this is insufficient - it only removes completely disconnected nodes, not low-connectivity ones.

### Proposed Implementation

Add filtering parameters to `FitnessComplexity` class:

```python
class FitnessComplexity:
    def __init__(
        self,
        n_iter: int = 200,
        tol: float = 1e-10,
        verbose: bool = True,
        min_ubiquity: int = 2,  # NEW: Filter products
        min_diversification: int = 5,  # NEW: Filter countries
        iterative_filter: bool = True  # NEW: Iteratively reapply filters
    ):
        ...
```

Filtering logic in `fit()`:
```python
def fit(self, M):
    """Fit Fitness-Complexity with connectivity filtering."""
    M_filtered = M.copy()
    
    if self.iterative_filter:
        # Iteratively filter until stable
        prev_shape = (-1, -1)
        while M_filtered.shape != prev_shape:
            prev_shape = M_filtered.shape
            
            # Filter low-diversification countries
            diversification = np.asarray(M_filtered.sum(axis=1)).ravel()
            valid_countries = diversification >= self.min_diversification
            M_filtered = M_filtered[valid_countries, :]
            
            # Filter low-ubiquity products
            ubiquity = np.asarray(M_filtered.sum(axis=0)).ravel()
            valid_products = ubiquity >= self.min_ubiquity
            M_filtered = M_filtered[:, valid_products]
            
        # Store mapping back to original indices
        self.country_mask_ = valid_countries
        self.product_mask_ = valid_products
    
    # Run algorithm on filtered matrix
    F_filtered, Q_filtered = _fitness_complexity(M_filtered, ...)
    
    # Expand back to original dimensions (filtered nodes get NaN or 0)
    self.fitness_ = self._expand_array(F_filtered, self.country_mask_)
    self.complexity_ = self._expand_array(Q_filtered, self.product_mask_)
    ...
```

### Alternative Approaches Considered

1. **Add small epsilon to prevent exact zeros**: Doesn't address the fundamental connectivity issue.
2. **Alternative initialization**: Still converges to degenerate solution due to unique products.
3. **Regularization term**: Adds complexity without addressing root cause.
4. **Warn but don't filter**: Puts burden on user to diagnose numerical issues.

**Decision**: Filtering is the standard, well-understood approach in the literature.

## Implementation Plan

### Phase 1: Add Filtering Parameters (2-3 days)
- [ ] Add `min_ubiquity`, `min_diversification`, `iterative_filter` to `__init__`
- [ ] Implement filtering logic in `fit()`
- [ ] Store `country_mask_` and `product_mask_` attributes
- [ ] Handle expansion back to original dimensions

### Phase 2: Testing (2 days)
- [ ] Unit tests for filtering logic
- [ ] Test with 2000 and 2020 Atlas data
- [ ] Verify fitness distributions are reasonable (no single-country dominance)
- [ ] Test edge cases (empty matrix after filtering, etc.)

### Phase 3: Documentation (1 day)
- [ ] Update `FitnessComplexity` docstring with filtering parameters
- [ ] Add examples showing filtered vs unfiltered results
- [ ] Document typical threshold values (ubiquity ≥ 2, diversification ≥ 5)

### Phase 4: Update Notebooks (1 day)
- [ ] Update `atlas_fitness_comparison.ipynb` to use filtering
- [ ] Show diagnostic plots comparing filtered vs unfiltered
- [ ] Document the impact on results

## Backward Compatibility

### Breaking Changes

Default behavior will change:
- **Before**: Run on raw matrix (numerical collapse possible)
- **After**: Filter by default (`min_ubiquity=2`, `min_diversification=5`)

### Migration Path

Users who want old behaviour can disable filtering:
```python
fc = FitnessComplexity(min_ubiquity=1, min_diversification=1)
```

Or explicitly:
```python
fc = FitnessComplexity(iterative_filter=False)
```

## Testing Strategy

### Unit Tests

```python
def test_fitness_filtering():
    """Test that low-connectivity nodes are filtered."""
    # Create matrix with unique product
    M = np.zeros((10, 10))
    M[0, 0] = 1  # Unique product for country 0
    M[1:, 1:] = np.random.randint(0, 2, (9, 9))  # Rest is random
    
    fc = FitnessComplexity(min_ubiquity=2)
    F, Q = fc.fit_transform(M)
    
    # Check that product 0 was filtered (ubiquity=1)
    assert fc.product_mask_[0] == False
    
    # Check that fitness is not dominated by country 0
    assert F[0] < 10 * np.median(F[F > 0])

def test_fitness_atlas_2020():
    """Test that 2020 Atlas data gives reasonable results."""
    M, countries, products = load_atlas_trade(year=2020, ...)
    
    fc = FitnessComplexity(min_ubiquity=2, min_diversification=5)
    F, Q = fc.fit_transform(M)
    
    # Check no single-country dominance
    max_fitness = F.max()
    median_fitness = np.median(F[F > 0])
    assert max_fitness / median_fitness < 20  # Reasonable ratio
    
    # Check that most countries have positive fitness
    n_nonzero = (F > 1e-10).sum()
    assert n_nonzero > 0.8 * len(F)
```

### Integration Tests

- Compare filtered results to R package with similar thresholds
- Test on nested matrices (should not over-filter)
- Test on sparse matrices (should filter aggressively)

## Related Requirements

None yet defined. This is a bug fix addressing a fundamental numerical issue.

## Implementation Status

- [ ] Design complete
- [ ] Filtering parameters added to `__init__`
- [ ] Filtering logic implemented in `fit()`
- [ ] Mask attributes added (`country_mask_`, `product_mask_`)
- [ ] Expansion logic for filtered results
- [ ] Unit tests written
- [ ] Integration tests with Atlas data
- [ ] Documentation updated
- [ ] Notebooks updated
- [ ] Validation against R package

## References

1. Tacchella, A., Cristelli, M., Caldarelli, G., Gabrielli, A., & Pietronero, L. (2012). "A New Metrics for Countries' Fitness and Products' Complexity". *Scientific Reports* 2:723. DOI: 10.1038/srep00723
   - Section on data preprocessing and connectivity requirements

2. Cristelli, M., Gabrielli, A., Tacchella, A., Caldarelli, G., & Pietronero, L. (2013). "Measuring the Intangibles: A Metrics for the Economic Complexity of Countries and Products". *PLoS ONE* 8(8): e70726.
   - Discusses filtering strategies for trade data

3. R `economiccomplexity` package source code:
   - https://github.com/pachamaltese/economiccomplexity
   - See `complexity_measures()` function for filtering approach

4. Lawrence, N.D. (2026). "Conditional Likelihood Interpretation of Economic Fitness". Working paper.
   - Theoretical justification for connectivity requirements

## Notes

### Why Filtering is Essential

Unlike linear methods (ECI via eigenvalues), the nonlinear Fitness-Complexity algorithm is **not robust to low-connectivity nodes**. The fixed-point iteration amplifies initial advantages from unique products, leading to runaway concentration.

### Recommended Thresholds

Based on literature review:
- **Conservative**: `min_ubiquity=5`, `min_diversification=10`
- **Moderate**: `min_ubiquity=2`, `min_diversification=5` (proposed default)
- **Minimal**: `min_ubiquity=2`, `min_diversification=2`

Thresholds depend on:
- Data density (sparser → more aggressive filtering)
- Number of entities (larger → can afford higher thresholds)
- Research question (structural analysis → conservative, rankings → moderate)

### Impact on Results

Filtering will:
- Remove isolated or weakly-connected nodes
- Produce more stable, interpretable fitness scores
- Reduce sensitivity to data artifacts (coding errors, niche products)
- Better align with theoretical assumptions

Trade-offs:
- Lose information about filtered entities
- May remove genuinely distinctive specializations
- Requires choosing threshold parameters
