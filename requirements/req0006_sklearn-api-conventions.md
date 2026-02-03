---
id: "0006"
title: "Algorithms follow scikit-learn estimator conventions"
status: "Proposed"
priority: "Medium"
created: "2026-02-02"
last_updated: "2026-02-02"
related_tenets:
  - "notebook-as-narrative-backed-by-functions"
stakeholders:
  - "Neil Lawrennd"
tags:
  - "api-design"
  - "usability"
  - "sklearn"
---

# REQ-0006: Algorithms follow scikit-learn estimator conventions

## Description

Core algorithms (Fitness-Complexity, ECI/PCI, Sinkhorn) should follow the scikit-learn estimator pattern: classes with `fit()`, `transform()`, and/or `predict()` methods. This makes algorithms more composable, intuitive for Python users, and compatible with scikit-learn utilities (pipelines, cross-validation, hyperparameter search).

Hyperparameters (e.g., `n_iter`, `tol`) are passed to `__init__()`; data (matrices) are passed to `fit()`.

## Acceptance Criteria

- [ ] Core algorithms are implemented as classes (e.g., `FitnessComplexity`, `ECI`, `SinkhornScaler`)
- [ ] Classes follow the estimator pattern:
  - Hyperparameters passed to `__init__(self, n_iter=200, tol=1e-10, ...)`
  - Data passed to `fit(self, X)` or `fit(self, X, y=None)`
  - Results accessible via `transform()`, `predict()`, or fitted attributes (e.g., `self.fitness_`, `self.complexity_`)
- [ ] Classes expose a `fit_transform(X)` convenience method for chaining
- [ ] Fitted attributes follow the trailing underscore convention (e.g., `self.fitness_`, `self.history_`)
- [ ] Classes are compatible with basic scikit-learn utilities (no need for full compliance, but pipelines and simple composition should work)
- [ ] Functional APIs remain available as convenience wrappers for backward compatibility and one-off usage
- [ ] Documentation shows both class-based and functional usage

## Notes (Optional)

The goal is **usability and composability**, not strict scikit-learn compliance. We don't need to implement `get_params()`/`set_params()` or full sklearn validation unless there's a clear use case.

Functional APIs (e.g., `fitness_complexity(M)`) can remain as convenience wrappers that internally instantiate and call the class methods.

## References

- **Related Tenets**: notebook-as-narrative-backed-by-functions
- **scikit-learn API**: https://scikit-learn.org/stable/developers/develop.html
