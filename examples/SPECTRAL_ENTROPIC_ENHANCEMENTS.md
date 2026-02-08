# Spectral-Entropic Comparison Notebook Enhancements

## Overview

The `spectral_entropic_comparison.ipynb` notebook has been enhanced to address two key limitations:

1. **Rigid diagnostic thresholds** → **Data-driven qualitative patterns**
2. **No community analysis** → **Eigenvector-based community detection and within-community analysis**

## Key Enhancements

### 1. Community Detection (community_analysis_helpers.py)

**New capability**: Detect sub-communities using higher eigenvectors and analyze ECI vs Fitness separately within each community.

#### detect_communities_from_eigenvectors()
- Uses k-means clustering on eigenvector embedding (ψ₂, ψ₃, ..., ψₖ)
- Automatic community number detection via **eigengap heuristic**
  - Converts transition eigenvalues λᵢᵀ to Laplacian eigenvalues λᵢᴸ = 1 - λᵢᵀ
  - Finds largest gap in Laplacian spectrum
  - Uses gap significance threshold to avoid over-fitting
- Returns community labels for each country

#### analyze_within_communities()
- For each detected community:
  - Extracts sub-network (countries + their products)
  - Computes ECI and Fitness within that sub-network
  - Calculates correlation within community
- Compares global vs within-community correlations

**Key insight**: For Morphology B (modular networks):
- Global correlation may be low (r < 0.5)
- Within-community correlations can be high (r > 0.8)
- This confirms: ECI identifies community boundaries, Fitness ranks within communities

### 2. Data-Driven Diagnostics (replacing rigid thresholds)

**Old approach**:
```python
if pearson_countries > 0.85 and gap_ratio_c > 2:
    morphology = "A: Nested"
elif conductance < 0.15 and gap_ratio_c > 2:
    morphology = "B: Modular"
# etc.
```

**New approach**:
```python
# Report observed values
print(f"Correlation: {pearson_countries:.3f}")
print(f"Spectral gap: {gap_ratio_c:.3f}")
print(f"Conductance: {conductance:.3f}")

# Provide qualitative interpretation
if pearson_countries > 0.85:
    print("→ Very high: likely tight monotone trend")
elif pearson_countries > 0.6:
    print("→ Moderate: some alignment with possible structure")
# etc.
```

**Why this is better**:
- Real networks rarely fit clean categories
- Avoids false confidence from arbitrary thresholds
- Encourages visual inspection and domain knowledge
- More transparent about uncertainty

### 3. Enhanced Visualizations

**Community detection plots**:
- Communities colored in ECI-Fitness space
- Eigenvalue spectrum with gaps marked
- Shows which eigengaps were used for clustering

**Key metrics displayed**:
- Global correlation
- Per-community correlations
- Average within-community correlation
- Comparison highlighting when within >> global

## Usage Example

```python
# Run the standard analysis
M_modular, name_modular = generate_modular_network(n_countries=50, n_products=80, n_communities=2)
results_modular = analyze_and_plot(M_modular, name_modular)

# Now detect and analyze communities
analyze_network_with_communities(
    M_modular,
    name_modular,
    results_modular['eci'],
    results_modular['fitness'],
    results_modular['eigenvalues_c']
)
```

**Expected output for modular network**:
```
Community Detection Analysis: Modular (Low Conductance)
────────────────────────────────────────────────────────

Detected 2 communities

Community sizes:
  Community 0: 25 countries (50.0%)
  Community 1: 25 countries (50.0%)

Correlation Comparison (Global vs Within-Community)
────────────────────────────────────────────────────────────
Global correlation:                r =  0.423

Within-community correlations:
  Community 0 (n=25):  r =  0.876 (p=1.234e-08)
  Community 1 (n=25):  r =  0.891 (p=5.678e-09)

Average within-community:          r =  0.884

Key Insight:
────────────────────────────────────────────────────────────
✓ Within-community correlations are MUCH HIGHER than global
  → ECI identifies community boundaries (block labels)
  → Fitness measures capability within each community
  → This is Morphology B: complementary perspectives!
```

## Implementation Notes

### Eigengap Heuristic

The eigengap heuristic is a principled way to choose the number of communities:

1. Compute Laplacian eigenvalues: λᴸᵢ = 1 - λᵢᵀ
2. Compute gaps: Δᵢ = λᴸᵢ₊₁ - λᴸᵢ for i ≥ 1 (skip trivial eigenvalue)
3. Find largest gap: k* = argmax Δᵢ
4. Number of communities: k = k* + 1

**Why this works**: 
- In networks with k communities, the first k eigenvalues are small (nearly 0)
- A large gap appears between λᴸₖ and λᴸₖ₊₁
- This is analogous to the "elbow" in k-means clustering

### Within-Community Analysis

For each community:
1. Extract sub-matrix: keep only rows (countries) in that community
2. Filter products: keep only products exported by community members
3. Compute ECI and Fitness on sub-network
4. Calculate correlation

**Advantages**:
- Properly accounts for community-specific product space
- Avoids artifacts from cross-community connections
- Validates complementarity hypothesis at finer granularity

## Theoretical Foundations

### Higher Eigenvectors and Multi-Scale Structure

From the paper's discussion:

- **Morphology A (Nested)**: Single gradient captured by ψ₂ (Fiedler vector)
- **Morphology B (Modular)**: ψ₂ identifies primary split, ψ₃, ψ₄ identify sub-communities
- **Morphology D (Multi-Scale)**: Small spectral gaps indicate need for multiple coordinates

**Key insight**: When gap_ratio_c < 2, the 1D embedding (ECI) is insufficient. Higher eigenvectors (ψ₃, ψ₄, ...) are needed for a full representation.

### Complementarity in Modular Networks

**Global perspective** (low correlation):
- ECI: Countries sort by community membership
- Fitness: Countries sort by capability regardless of community

**Within-community perspective** (high correlation):
- Both ECI and Fitness agree on capability ranking
- Only meaningful when communities are separated

**Practical implication**: For real economic networks with regional communities (e.g., ASEAN, EU), global ECI vs Fitness discordance may indicate:
- ECI captures regional economic blocs
- Fitness captures absolute capability independent of region
- Within-region analysis shows both methods agree on capability gradients

## Files Modified

1. `community_analysis_helpers.py` - New file with community detection functions
2. `spectral_entropic_comparison.ipynb` - Enhanced notebook:
   - Added community analysis section (cells 23-25)
   - Updated morphology interpretation to be data-driven
   - Revised diagnostic guidelines to be qualitative
   - Added imports for community analysis

## Dependencies Added

- `sklearn.cluster.KMeans` - For eigenvector clustering
- `scipy.sparse.linalg.eigs` - Already used, now also in community detection

## Future Enhancements

Potential additions:
1. **Robustness analysis**: Bootstrap resampling to assess community detection stability
2. **Hierarchical communities**: Recursive application to detect nested structure
3. **Product-side communities**: Analogous analysis for product space (via ψ₂ᴾ, ψ₃ᴾ, ...)
4. **Directed graphs**: Extend to directed trade networks using eigenvectors of non-symmetric matrices
5. **Temporal evolution**: Track how communities and correlations change over time

## References

- Eigengap heuristic: von Luxburg, U. (2007). "A tutorial on spectral clustering." Statistics and computing 17(4), 395-416.
- Multi-scale structure: Newman, M. E. (2006). "Modularity and community structure in networks." PNAS 103(23), 8577-8582.
- Economic complexity: Hidalgo & Hausmann (2009), Tacchella et al. (2012), Balland & Rigby (2016)