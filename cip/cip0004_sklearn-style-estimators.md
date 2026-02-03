---
id: "0004"
title: "Refactor algorithms into scikit-learn-style estimators"
status: "Proposed"
created: "2026-02-02"
last_updated: "2026-02-02"
author: "Neil Lawrennd"
compressed: false
related_requirements:
  - "0006"
tags:
  - "api-design"
  - "refactoring"
  - "sklearn"
---

# CIP-0004: Refactor algorithms into scikit-learn-style estimators

> **Note**: This CIP describes HOW to achieve REQ-0006 (scikit-learn API conventions).

## Status

- [x] Proposed
- [ ] Accepted
- [ ] In Progress
- [ ] Implemented
- [ ] Closed

## Summary

Refactor core algorithms (Fitness-Complexity, ECI/PCI, Sinkhorn) from functional APIs into scikit-learn-style estimator classes with `fit()`, `transform()`, and fitted attributes. Maintain backward compatibility by keeping functional wrappers.

## Motivation

The current functional API (`fitness_complexity(M, n_iter=200)`) is:
- Less composable: cannot be chained or pipelined easily
- Less familiar: Python users expect the fit/transform pattern from scikit-learn
- Less convenient: cannot inspect fitted state after computation

A class-based API (`FitnessComplexity(n_iter=200).fit(M)`) enables:
- **Composition**: Use in scikit-learn pipelines, cross-validation, hyperparameter search
- **Familiarity**: Follows established Python scientific stack conventions
- **Inspectability**: Fitted attributes (`self.fitness_`, `self.history_`) available for introspection
- **Reuse**: Fit once, transform multiple times (e.g., for Sinkhorn scaling)

## Detailed Description

### Proposed API

**FitnessComplexity**:
```python
from fitkit.algorithms import FitnessComplexity

# Class-based (recommended)
fc = FitnessComplexity(n_iter=200, tol=1e-10)
fc.fit(M)  # Computes fitness and complexity
F = fc.fitness_  # Fitted country/user fitness scores
Q = fc.complexity_  # Fitted product/word complexity scores
history = fc.history_  # Convergence history

# Or: one-liner
F, Q = FitnessComplexity(n_iter=200).fit_transform(M)

# Functional API (backward compat, convenience)
from fitkit.algorithms.fitness import fitness_complexity
F, Q, history = fitness_complexity(M, n_iter=200, tol=1e-10)
```

**ECI**:
```python
from fitkit.algorithms import ECI

# Class-based
eci_estimator = ECI()
eci_estimator.fit(M)
eci = eci_estimator.eci_  # Country ECI scores
pci = eci_estimator.pci_  # Product PCI scores

# Or: one-liner
eci, pci = ECI().fit_transform(M)

# Functional API (backward compat)
from fitkit.algorithms.eci import compute_eci_pci
eci, pci = compute_eci_pci(M)
```

**SinkhornScaler**:
```python
from fitkit.algorithms import SinkhornScaler

# Class-based (most useful here: fit once, transform multiple times)
scaler = SinkhornScaler(n_iter=2000, tol=1e-12)
scaler.fit(M, row_marginals=r, col_marginals=c)
W = scaler.transform(M)  # Apply fitted scaling
# Or: W = scaler.W_  # Scaled matrix as fitted attribute

# Functional API (backward compat)
from fitkit.algorithms.sinkhorn import sinkhorn_masked
u, v, W, history = sinkhorn_masked(M, r, c, n_iter=2000, tol=1e-12)
```

### Implementation Strategy

1. **Create estimator classes** in existing modules:
   - `fitkit/algorithms/fitness.py`: Add `FitnessComplexity` class
   - `fitkit/algorithms/eci.py`: Add `ECI` class
   - `fitkit/algorithms/sinkhorn.py`: Add `SinkhornScaler` class

2. **Estimator structure**:
   ```python
   class FitnessComplexity:
       def __init__(self, n_iter=200, tol=1e-10):
           self.n_iter = n_iter
           self.tol = tol
       
       def fit(self, X, y=None):
           """Compute fitness-complexity on binary incidence matrix X."""
           self.fitness_, self.complexity_, self.history_ = fitness_complexity(
               X, n_iter=self.n_iter, tol=self.tol
           )
           return self
       
       def fit_transform(self, X, y=None):
           """Fit and return (fitness, complexity)."""
           self.fit(X, y)
           return self.fitness_, self.complexity_
   ```

3. **Keep functional APIs** as module-level functions for backward compatibility

4. **Update `__init__.py`** to export classes:
   ```python
   from fitkit.algorithms.fitness import FitnessComplexity, fitness_complexity
   from fitkit.algorithms.eci import ECI, compute_eci_pci
   from fitkit.algorithms.sinkhorn import SinkhornScaler, sinkhorn_masked
   ```

5. **Update tests** to cover both class-based and functional APIs

6. **Update documentation** to show class-based usage as primary, functional as convenience

### Backward Compatibility

**Fully backward compatible**: Existing functional APIs remain unchanged. Users can migrate incrementally or continue using functional APIs indefinitely.

Old code continues to work:
```python
F, Q, history = fitness_complexity(M, n_iter=200)
```

New code can use class-based API:
```python
F, Q = FitnessComplexity(n_iter=200).fit_transform(M)
```

## Implementation Plan

1. **Phase 1**: Add estimator classes alongside existing functional APIs
   - Implement `FitnessComplexity`, `ECI`, `SinkhornScaler`
   - Internal methods delegate to existing functional implementations
   - Export classes from `__init__.py`

2. **Phase 2**: Add tests for estimator classes
   - Test `fit()`, `transform()`, `fit_transform()` methods
   - Test fitted attributes (`fitness_`, `complexity_`, etc.)
   - Verify backward compatibility of functional APIs

3. **Phase 3**: Update documentation and examples
   - Update docstrings to show class-based usage first
   - Add examples comparing functional vs. class-based APIs
   - Update notebook (when refactored) to use class-based API

## Testing Strategy

- Add tests for each estimator class (fit, transform, fitted attributes)
- Verify functional APIs still work (backward compatibility)
- Test composability: simple pipelines, method chaining
- Do NOT require full sklearn compliance (get_params/set_params)

## Related Requirements

This CIP implements:
- **REQ-0006**: Algorithms follow scikit-learn estimator conventions

## Implementation Status

- [ ] Create estimator classes (FitnessComplexity, ECI, SinkhornScaler)
- [ ] Add tests for estimator classes
- [ ] Update __init__.py exports
- [ ] Update documentation

## References

- scikit-learn API: https://scikit-learn.org/stable/developers/develop.html
- Related tenet: notebook-as-narrative-backed-by-functions
