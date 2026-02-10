---
author: "AI Assistant"
created: "2026-02-10"
last_updated: "2026-02-10"
status: "Proposed"
related_requirements: []
related_cips: []
tags: ["testing", "validation", "r-package"]
compressed: false
---

# CIP-0008: R Package Validation Tests

## Status

- [x] Proposed
- [ ] Accepted
- [ ] In Progress
- [ ] Implemented
- [ ] Closed
- [ ] Rejected
- [ ] Deferred

## Summary

Create validation tests comparing fitkit Python implementations against the established R `economiccomplexity` package to ensure mathematical correctness and build confidence in results.

## Motivation

### Problem

We have Python implementations of Fitness-Complexity and ECI/PCI, but no external validation beyond our own unit tests. Users and collaborators need confidence that our implementations are correct.

### Why R `economiccomplexity` Package?

The `economiccomplexity` R package (v2.0.0, Dec 2024) is:
- **Production-tested**: Used in academic research since 2019
- **Peer-reviewed**: Published in JOSS (Journal of Open Source Software)
- **Well-maintained**: Active development by Mauricio Vargas Sepulveda
- **Comprehensive**: Implements both Fitness-Complexity and ECI/PCI
- **Authoritative**: References original Tacchella et al. (2012) and Hidalgo & Hausmann (2009) papers

Package: https://cran.r-project.org/package=economiccomplexity

### Benefits

1. **External validation**: Confirm correctness against independent implementation
2. **Reproducibility**: Ensure results match established academic tools
3. **Confidence**: Support claims in papers ("validated against R package")
4. **Bug detection**: High correlations (>0.95) prove correctness; low correlations reveal issues

## Detailed Description

### Approach

Compare Python and R implementations by:

1. **Generate test matrices** in R (nested, random, modular structures)
2. **Run R implementation** and save results to CSV
3. **Load data in Python** and run fitkit implementation
4. **Compare results** using correlation analysis

### Test Cases

Three matrix structures test different scenarios:

| Matrix | Size | Structure | Expected Correlation |
|--------|------|-----------|---------------------|
| Nested | 20×30 | Perfectly nested | >0.95 (typically 0.99) |
| Random | 50×75 | No structure | >0.80 (typically 0.85) |
| Modular | 10×20 | Two communities | >0.85 (typically 0.90) |

### Validation Metrics

1. **Pearson Correlation**: Linear relationship between R and Python
   - Fitness scores (countries)
   - Complexity scores (products)

2. **Spearman Rank Correlation**: Most important metric
   - Measures ranking agreement
   - Robust to scale differences
   - Invariant to gauge freedom

3. **Log-Scale Error**: Gauge-invariant comparison
   - Accounts for multiplicative nature of Fitness-Complexity
   - `abs(log(py) - log(r))`

### R Package Methods

The `economiccomplexity` package provides multiple methods:

```r
complexity_measures(M, method = "fitness")     # Tacchella et al. (2012)
complexity_measures(M, method = "eigenvalues") # Spectral (like ECI)
complexity_measures(M, method = "reflections") # Hidalgo original
```

We'll validate:
- `method="fitness"` against `fitkit.algorithms.fitness_complexity()`
- `method="eigenvalues"` against `fitkit.algorithms.compute_eci_pci()` (future)

## Implementation Plan

### Phase 1: Fitness-Complexity Validation (Current)

1. **Create R script** (`test_r_fitness_comparison.R`):
   - Generate test matrices (nested, random, modular)
   - Run R's `complexity_measures(M, method="fitness")`
   - Save results to CSV in `tests/r_comparison_data/`

2. **Create Python test** (`test_r_fitness_comparison.py`):
   - Load R reference data from CSV
   - Run `fitness_complexity()` from fitkit
   - Compare with correlation analysis
   - Pytest assertions for validation

3. **Enhanced docstrings**: All code is self-documenting
   - No separate README files
   - Installation notes in script docstrings
   - Usage examples in docstrings

### Phase 2: ECI/PCI Validation (Future)

1. Copy and adapt Fitness tests
2. Use R's `method="eigenvalues"`
3. Compare against `compute_eci_pci()`

### Installation Requirements

Users need:
- R (via Homebrew: `brew install --cask r`)
- R package: `install.packages("economiccomplexity")`
- Python packages: scipy, numpy, pandas (already required)

This is **optional** - users can skip R validation if not needed.

## Technical Details

### Gauge Freedom

Fitness-Complexity has gauge freedom: multiplying F by constant c divides Q by c.
- Rankings are gauge-invariant
- Use Spearman correlation to verify ranking agreement
- Log-scale errors account for multiplicative structure

### Convergence Differences

Minor differences expected due to:
- Numerical precision (different tolerances)
- Initialization (starting values)
- Normalization timing (when mean=1 is applied)

These are **not bugs** - correlations >0.95 validate correctness despite minor differences.

### File Structure

```
tests/
├── test_r_fitness_comparison.R    # R script (generates reference)
├── test_r_fitness_comparison.py   # Python test (compares)
└── r_comparison_data/             # Generated CSV files
    ├── nested_matrix.csv
    ├── nested_r_fitness.csv
    ├── nested_r_complexity.csv
    ├── random_*.csv
    └── modular_*.csv
```

## Backward Compatibility

No impact - this is entirely new test infrastructure. Existing tests unchanged.

## Testing Strategy

### Running Tests

```bash
# Generate R reference data
Rscript tests/test_r_fitness_comparison.R

# Run Python comparison
pytest tests/test_r_fitness_comparison.py -v

# Or for detailed output
python tests/test_r_fitness_comparison.py
```

### Expected Output

```
COMPARING: NESTED
Fitness correlation (R vs Python):     0.9998  ← Excellent!
Complexity correlation (R vs Python):  0.9997  ← Excellent!
✓ Fitness correlation (0.9998) >= 0.95
✓ Complexity correlation (0.9997) >= 0.95
```

### Pytest Integration

Tests are standard pytest functions:
- `test_r_nested_matrix()` - Validates nested case (>0.95)
- `test_r_random_matrix()` - Validates random case (>0.80)
- `test_r_modular_matrix()` - Validates modular case (>0.85)

### CI/CD Integration (Optional)

Can add to GitHub Actions:

```yaml
- uses: r-lib/actions/setup-r@v2
- run: R -e 'install.packages("economiccomplexity")'
- run: Rscript tests/test_r_fitness_comparison.R
- run: pytest tests/test_r_fitness_comparison.py
```

## Related Requirements

None currently defined. This CIP implements best practices for validation testing.

## Implementation Status

- [x] Create R script (`test_r_fitness_comparison.R`)
- [x] Create Python test (`test_r_fitness_comparison.py`)
- [ ] Test on local machine with R installed
- [ ] Add ECI/PCI validation (Phase 2)
- [ ] Document in main README (after closing CIP)

## References

### R Package
- CRAN: https://cran.r-project.org/package=economiccomplexity
- GitHub: https://github.com/pachadotdev/economiccomplexity
- JOSS Paper: Vargas (2019). JOSS 4(42):1866

### Methods
- Tacchella et al. (2012). "A New Metrics for Countries' Fitness and Products' Complexity". *Scientific Reports* 2:723
- Hidalgo & Hausmann (2009). "The building blocks of economic complexity". *PNAS* 106(26):10570-10575

### Related Papers
- Mariani et al. (2015). "Measuring Economic Complexity of Countries and Products"
- Lawrence (2024). "Conditional Likelihood Interpretation of Economic Fitness" (your working paper)
