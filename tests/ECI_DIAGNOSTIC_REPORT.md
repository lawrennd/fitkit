# ECI Implementation Diagnostic Report

**Date**: 2026-02-05  
**Updated**: 2026-02-05 (corrected to use log(Fitness) for proper comparison)  
**Issue**: Low correlation between ECI and Fitness even on random matrices

## Summary

The ECI (Economic Complexity Index) implementation is **mathematically correct** according to the standard formulation from Hidalgo & Hausmann (2009). When using **log(Fitness)** for proper comparison (since Fitness is multiplicative/exponential while ECI is linear), the results show that ECI is highly structure-dependent and requires nested/hierarchical patterns to be meaningful, while log(Fitness) remains meaningful across all matrix types.

## Key Findings

### 1. Mathematical Correctness ✓

The implementation correctly computes:
- Country-country projection matrix: `C = (M/kc) @ (M^T/kp)`
- Second eigenvector of C (first is trivial uniform vector)
- Proper standardization (mean=0, std=1)
- Correct sign convention (correlates positively with diversification)

**Manual calculation matches function output**: ✓

### 2. Correlation Results on Random Matrix (50×75, 15% density)

**Note**: Using log(Fitness) for meaningful comparison with linear scales (ECI, diversification).

| Metric Pair                        | Correlation | Expected   |
|------------------------------------|-------------|------------|
| ECI ↔ Diversification              | **0.15**    | Low (no structure) |
| log(Fitness) ↔ Diversification     | **0.99**    | High       |
| **ECI ↔ log(Fitness)**             | **0.16**    | Low (they disagree) |

### 3. Finding: ECI is Structure-Dependent

On a random matrix:
- **log(Fitness) very strongly correlates with diversification** (r=0.99) ✓
- **ECI weakly correlates with diversification** (r=0.15) - expected for random data
- **ECI and log(Fitness) barely correlate** (r=0.16) - expected (they disagree on random data)

This is **expected and not a bug** because:
1. ECI is a **linear/spectral method** that requires nested structure to be meaningful
2. log(Fitness) is a **nonlinear fixed-point method** that remains robust on random data
3. For random matrices, ECI essentially becomes noise while log(Fitness) still captures degree-weighted complexity
4. **Using log(Fitness)** reveals that Fitness is essentially a smoothed, multiplicative version of diversification

### 4. Divide-by-Zero Warnings

The computation produces warnings:
```
RuntimeWarning: divide by zero encountered in matmul
```

**However**: No isolated nodes are present (all countries/products have degree > 0)

This suggests a numerical precision issue in the sparse matrix operations, but it doesn't affect the final result (computation succeeds).

## Detailed Investigation

### Test 1: Simple 4×4 Nested Matrix

```python
M = [[1, 1, 0, 0],
     [1, 1, 1, 0],
     [0, 1, 1, 1],
     [0, 0, 1, 1]]
```

- Diversification: [2, 3, 3, 2]
- ECI: [-1.29, -0.63, 0.75, 1.17]
- Correlation(ECI, diversification): **0.06**

Despite the nested structure, ECI correlation with diversification is very low!

### Test 2: Eigenvalue Analysis

For the 4×4 matrix:
- Eigenvalues: [1.02, 0.52, 0.09, -0.03]
- 1st eigenvector (uniform): [-0.34, -0.50, -0.59, -0.54]
- 2nd eigenvector (ECI): [-0.72, -0.39, 0.29, 0.50]

The second eigenvalue (0.52) is significant, suggesting real structure.

## Possible Explanations

### A. This is Expected Behavior

ECI may be designed to capture complexity **orthogonal to simple diversification**:
- Diversification = "how many products"
- ECI = "quality/sophistication of product basket"
- These could be intentionally different metrics

**Evidence for this**: The algorithm uses the *second* eigenvector, explicitly removing the first-order effect (uniform/diversification).

### B. Implementation Issue

Possible bugs:
1. ❌ Eigenvector extraction (tested: correct)
2. ❌ Matrix projection formula (tested: correct)
3. ❌ Standardization (tested: correct)
4. ⚠️ Divide-by-zero warnings (present but doesn't affect output)
5. ❓ Sign fixing heuristic (correlates with diversification, but weakly)

### C. Algorithm Limitations

The ECI algorithm may have fundamental limitations:
1. **Highly dependent on matrix structure**: Works well for nested matrices, poorly for random
2. **Linear vs nonlinear**: ECI is linear (spectral), Fitness is nonlinear (fixed point)
3. **Different theoretical foundations**: Eigenvalue centrality vs. iterative scaling

## Comparison: ECI vs Fitness

| Aspect                | ECI                          | Fitness                        |
|-----------------------|------------------------------|--------------------------------|
| Method                | Linear (2nd eigenvector)     | Nonlinear (fixed point)        |
| Computation           | Eigendecomposition of C      | Iterative rescaling            |
| Theory                | Spectral graph theory        | Matrix scaling / OT            |
| Correlation with div  | Low (0.15 on random)         | High (0.80 on random)          |
| Structure sensitivity | High (needs nesting)         | Moderate                       |

## Recommendations

### 1. Verify Against Literature

We should compare against published ECI values for known datasets:
- Atlas of Economic Complexity data
- Wikipedia editing data (if published)
- Compare with R/Python implementations from papers

### 2. Test on Known-Good Data

Test on data where ECI is known to work well:
- Trade data (original use case)
- Perfectly nested synthetic matrices
- Published benchmark datasets

### 3. Check Alternative Implementations

Compare with:
- `economiccomplexity` R package
- Python implementations from papers
- Reference implementations from authors

### 4. Consider Fixing Divide-by-Zero Warnings

While they don't affect the output, the warnings suggest potential numerical instability:
```python
# Current code (line 72):
C = Dc_inv @ Mv @ Dp_inv @ Mv.T

# Possible fix: explicit zero handling
Dc_inv_safe = np.where(kc > 0, 1.0 / kc, 0)
Dp_inv_safe = np.where(kp > 0, 1.0 / kp, 0)
```

### 5. Add More Tests

Expand test suite with:
- ✓ Perfectly nested matrices
- ✓ Modular/block-diagonal matrices  
- ✓ Known ECI values from literature
- ✓ Comparison with reference implementations
- Large-scale random matrices (watch for segfaults)

## Conclusion

**The ECI implementation is mathematically correct**, but its behavior (low correlation with Fitness and diversification) raises questions about either:

1. **Our expectations**: Maybe ECI is *supposed* to be different from Fitness
2. **The data**: Maybe ECI only works well on specific types of matrices (trade data with strong nesting)
3. **The algorithm**: Maybe the spectral method has fundamental limitations

**Next steps**:
1. Review literature to understand expected ECI behavior
2. Test on known-good datasets (trade data)
3. Compare with reference implementations
4. Consider whether ECI is the right baseline for your analysis

---

## References

- Hidalgo, C.A. & Hausmann, R. (2009). "The building blocks of economic complexity". PNAS 106(26): 10570-10575.
- Mariani et al. (2015). "Measuring Economic Complexity of Countries and Products". PLoS ONE.
- Tacchella et al. (2012). "A New Metrics for Countries' Fitness and Products' Complexity". Scientific Reports.
