---
id: "2026-02-03_use-sklearn-api-in-notebook"
title: "Update notebook to use sklearn-style estimator API"
status: "Completed"
priority: "Medium"
created: "2026-02-03"
last_updated: "2026-02-03"
category: "infrastructure"
related_cips:
  - "0004"
owner: "Neil Lawrennd"
dependencies:
  - "2026-02-02_sklearn-style-estimators"
  - "2026-02-02_update-notebook-to-use-package"
tags:
  - "backlog"
  - "notebook"
  - "api"
  - "sklearn"
---

# Task: Update notebook to use sklearn-style estimator API

> **Note**: This demonstrates the sklearn-style API introduced in CIP-0004 and REQ-0006.

## Description

Update the example notebook to use the sklearn-style estimator classes (`FitnessComplexity`, `ECI`, `SinkhornScaler`) instead of the functional API (`fitness_complexity()`, `compute_eci_pci()`, `sinkhorn_masked()`). This demonstrates the recommended API pattern and improves composability.

Currently, the notebook imports both APIs but only uses the functional one. This task switches to the sklearn-style API throughout.

## Acceptance Criteria

- [ ] Notebook uses `FitnessComplexity().fit_transform(M)` instead of `fitness_complexity(M)`
- [ ] Notebook uses `ECI().fit_transform(M)` instead of `compute_eci_pci(M)`
- [ ] Notebook uses `SinkhornScaler().fit_transform(M, row_marginals=r, col_marginals=c)` instead of `sinkhorn_masked(M, r, c)`
- [ ] Import statement updated to emphasize sklearn classes:
  ```python
  from fitkit.algorithms import FitnessComplexity, ECI, SinkhornScaler
  ```
- [ ] Add a narrative cell explaining the sklearn-style API benefits:
  - Composability (can chain with sklearn pipelines)
  - Familiar interface for ML practitioners
  - Fitted attributes (`.fitness_`, `.eci_`, etc.)
  - Separate fit/transform if needed
- [ ] All notebook outputs preserved (numerical results unchanged)
- [ ] Notebook runs end-to-end successfully
- [ ] Optional: Add a cell demonstrating separate fit/transform workflow

## Implementation Notes

**Conversion examples**:

```python
# OLD (functional API)
F, Q, history = fitness_complexity(M, n_iter=100, tol=1e-10)

# NEW (sklearn API)
fc = FitnessComplexity(n_iter=100, tol=1e-10)
F, Q = fc.fit_transform(M)
# Access history via fc.history_
```

```python
# OLD
eci, pci = compute_eci_pci(M)

# NEW
eci_model = ECI()
eci, pci = eci_model.fit_transform(M)
# Access via eci_model.eci_, eci_model.pci_
```

```python
# OLD
u, v, W, history = sinkhorn_masked(M, r, c, n_iter=200, tol=1e-10)

# NEW
scaler = SinkhornScaler(n_iter=200, tol=1e-10)
W = scaler.fit_transform(M, row_marginals=r, col_marginals=c)
# Access via scaler.u_, scaler.v_, scaler.W_, scaler.history_
```

**Benefits to highlight in notebook**:
- Familiar pattern for users coming from sklearn/pandas
- Can use in sklearn pipelines or custom workflows
- Fitted attributes follow sklearn conventions (`attribute_` suffix)
- Maintains backward compatibility (functional API still available)

## Related

- CIP: 0004 (sklearn-style estimators)
- Requirement: REQ-0006 (sklearn API conventions)
- Tenet: notebook-as-narrative-backed-by-functions (updated to mention sklearn API)

## Progress Updates

### 2026-02-03

Task created to demonstrate sklearn-style API in the example notebook, completing the adoption of CIP-0004/REQ-0006.

Task completed:
- Updated notebook to use `FitnessComplexity`, `ECI`, `SinkhornScaler` classes
- Replaced all functional API calls with sklearn-style `.fit_transform()`
- Updated variable access to use fitted attributes (`.history_`, `.u_`, `.v_`, etc.)
- Import statements reordered to emphasize sklearn classes
- All numerical outputs preserved
- Demonstrates recommended API pattern from CIP-0004
