---
id: "2026-02-02_extract-diagnostics-and-add-tests"
title: "Extract diagnostic utilities and add test suite"
status: "Proposed"
priority: "High"
created: "2026-02-02"
last_updated: "2026-02-02"
category: "infrastructure"
related_cips:
  - "0002"
owner: "Neil Lawrennd"
dependencies:
  - "2026-02-02_create-fitkit-package-structure"
tags:
  - "backlog"
  - "diagnostics"
  - "testing"
---

# Task: Extract diagnostic utilities and add test suite

> **Note**: This task implements CIP-0002 phases 4-5 (diagnostics, tests).

## Description

Extract visualization and diagnostic utilities from the notebook into reusable modules, and create a comprehensive test suite using pytest and synthetic fixtures. This enables REQ-0002 (testing and verification).

## Acceptance Criteria

- [ ] `fitkit/diagnostics/plots.py` contains extracted plotting functions:
  - `plot_circular_bipartite_flow()`
  - `plot_alluvial_bipartite()`
  - `plot_dual_potential_bipartite()`
  - `plot_ranked_barcodes()`
- [ ] `fitkit/diagnostics/comparison.py` contains helper to compute FC + ECI/PCI together and return aligned tables
- [ ] `tests/` directory exists with pytest configuration
- [ ] `tests/test_fitness.py` contains unit tests for Fitness-Complexity iteration:
  - Convergence on small fixtures
  - Scale-free gauge (mean normalization)
  - Edge cases (isolated nodes, degenerate support)
- [ ] `tests/test_eci.py` contains unit tests for ECI/PCI computation:
  - Known fixtures with expected outputs
  - Spectral properties
- [ ] `tests/test_sinkhorn.py` contains unit tests for masked Sinkhorn:
  - Convergence
  - Marginal matching
  - Sparsity preservation
- [ ] `tests/test_loaders.py` contains tests for data loaders using synthetic fixtures (no network/credentials)
- [ ] `pytest` runs successfully without network access or cloud credentials
- [ ] Test outcomes are deterministic

## Implementation Notes

**Test fixtures**:
- Create small synthetic matrices (5-10 rows/cols) with known properties
- Include edge cases: empty rows/cols, isolated nodes, fully connected graphs
- Store fixtures in `tests/fixtures.py` or `fitkit/data/fixtures.py`

**Test categories**:
- **Numerical correctness**: Validate against hand-computed or published results
- **Convergence**: Ensure iterative methods converge within expected iterations
- **Invariants**: Check structural properties (shapes, sparsity, gauge conditions)
- **Determinism**: Same inputs → same outputs

**pytest configuration** (in `pyproject.toml`):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
```

## Related

- CIP: 0002
- Requirements: REQ-0002 (testing)

## Progress Updates

### 2026-02-02

Task created to implement CIP-0002 phases 4-5 (diagnostics and tests).
