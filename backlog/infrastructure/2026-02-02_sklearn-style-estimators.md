---
id: "2026-02-02_sklearn-style-estimators"
title: "Refactor algorithms into scikit-learn-style estimators"
status: "Completed"
priority: "Medium"
created: "2026-02-02"
last_updated: "2026-02-02"
category: "infrastructure"
related_cips:
  - "0004"
owner: "Neil Lawrennd"
dependencies: []
tags:
  - "backlog"
  - "api-design"
  - "refactoring"
---

# Task: Refactor algorithms into scikit-learn-style estimators

> **Note**: This task implements CIP-0004 (sklearn-style estimators).

## Description

Refactor core algorithms (Fitness-Complexity, ECI/PCI, Sinkhorn) into scikit-learn-style estimator classes with `fit()`, `transform()`, and fitted attributes. Maintain full backward compatibility by keeping functional APIs as convenience wrappers.

## Acceptance Criteria

- [ ] `FitnessComplexity` class with `fit()`, `fit_transform()`, and fitted attributes (`fitness_`, `complexity_`, `history_`)
- [ ] `ECI` class with `fit()`, `fit_transform()`, and fitted attributes (`eci_`, `pci_`)
- [ ] `SinkhornScaler` class with `fit()`, `transform()`, `fit_transform()`, and fitted attributes (`u_`, `v_`, `W_`, `history_`)
- [ ] Functional APIs remain unchanged for backward compatibility
- [ ] Classes exported from `fitkit.algorithms.__init__.py`
- [ ] Tests updated to cover both class-based and functional APIs
- [ ] All existing tests pass (backward compatibility verified)

## Implementation Notes

**Implementation strategy**:
1. Add estimator classes in existing modules (fitness.py, eci.py, sinkhorn.py)
2. Estimators delegate to existing functional implementations (DRY)
3. Export classes from `__init__.py`
4. Add tests for estimator classes
5. Verify functional APIs still work

**Example implementation pattern**:
```python
class FitnessComplexity:
    def __init__(self, n_iter=200, tol=1e-10):
        self.n_iter = n_iter
        self.tol = tol
    
    def fit(self, X, y=None):
        self.fitness_, self.complexity_, self.history_ = fitness_complexity(
            X, n_iter=self.n_iter, tol=self.tol
        )
        return self
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.fitness_, self.complexity_
```

## Related

- CIP: 0004
- Requirements: REQ-0006 (sklearn API conventions)

## Progress Updates

### 2026-02-02

Task created to implement CIP-0004 (sklearn-style estimators).

Task completed:
- Added FitnessComplexity class with fit(), fit_transform(), and fitted attributes (fitness_, complexity_, history_, n_iter_)
- Added ECI class with fit(), fit_transform(), and fitted attributes (eci_, pci_)
- Added SinkhornScaler class with fit(), transform(), fit_transform(), and fitted attributes (u_, v_, W_, history_)
- All functional APIs remain unchanged (full backward compatibility)
- Updated fitkit.algorithms.__init__.py to export estimator classes
- Added 15 new tests for estimator classes (5 per algorithm)
- All existing tests pass (backward compatibility verified)
- Estimators follow sklearn conventions: hyperparameters in __init__, data in fit()
- Fitted attributes use trailing underscore convention (e.g., fitness_)
