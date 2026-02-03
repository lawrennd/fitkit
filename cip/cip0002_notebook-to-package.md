---
id: "0002"
title: "Convert notebook capabilities into a Python package"
status: "Accepted"
created: "2026-02-02"
last_updated: "2026-02-02"
author: "Neil Lawrennd"
compressed: false
related_requirements:
  - "0002"
  - "0003"
  - "0004"
tags:
  - "architecture"
  - "package-structure"
  - "scientific-software"
---

# CIP-0002: Convert notebook capabilities into a Python package

> **Note**: This CIP describes HOW to achieve REQ-0002 (testing), REQ-0003 (algorithm/data separation), and REQ-0004 (adapter interface).

## Status

- [x] Proposed
- [x] Accepted
- [ ] In Progress
- [ ] Implemented
- [ ] Closed

## Summary

Extract reusable code from `wikipedia_editing_fitness_complexity.ipynb` into a `fitkit/` Python package while keeping the notebook as the primary narrative. The package will provide testable, reusable modules for data loading, algorithms, and diagnostics that align with the scientific-software tenets and the stated architectural boundaries.

## Motivation

Currently, all data loading, algorithms, and diagnostics live inside the notebook. This makes it difficult to:
- Test algorithms independently (requires copy/paste or mock cells).
- Reuse code in other notebooks/scripts without duplication.
- Swap data sources (BigQuery vs local fixtures) without editing algorithm logic.
- Integrate with future data-oriented layers (e.g. Lynguine).

This CIP addresses REQ-0002 (testing), REQ-0003 (separation), and REQ-0004 (adapter interface) by defining a package structure that keeps concerns separate and boundaries explicit.

## Detailed Description

### Package Structure

Create a `fitkit/` directory with the following modules:

```
fitkit/
├── __init__.py                 # Package exports
├── data/
│   ├── __init__.py
│   ├── loaders.py              # Data-loading adapter interface + implementations
│   └── fixtures.py             # Synthetic data generators for testing
├── algorithms/
│   ├── __init__.py
│   ├── fitness.py              # Fitness-Complexity fixed point
│   ├── eci.py                  # ECI/PCI spectral computation
│   └── sinkhorn.py             # Masked Sinkhorn/IPF scaling
├── diagnostics/
│   ├── __init__.py
│   ├── plots.py                # Visualisation utilities (circular/alluvial/barcodes)
│   └── comparison.py           # Baseline comparison helpers (FC vs ECI)
└── types.py                    # Shared type hints and data structures
```

### Design Principles

**1. Algorithms accept in-memory inputs only**
- Functions like `fitness_complexity(M_bin, ...)`, `compute_eci_pci(M_bin)`, and `sinkhorn_masked(M_bin, r, c, ...)` will accept sparse matrices and arrays.
- No filesystem or network access in algorithm code.

**2. Data loading is isolated in `fitkit.data.loaders`**
- Define a `DataLoader` protocol/ABC with a single method:
  ```python
  def load() -> DataBundle:
      """Return in-memory representation: matrix, row_labels, col_labels, metadata."""
  ```
- Implement `WikipediaLoader` (BigQuery + caching), `ParquetLoader`, `SyntheticLoader`.
- Each loader encapsulates its own I/O logic and returns a standardized in-memory bundle.

**3. Fixtures for testing**
- `fitkit.data.fixtures` provides small synthetic matrices (edge cases: isolated nodes, empty columns, etc.).
- Fixtures are deterministic and fast.

**4. Diagnostics as reusable utilities**
- Plotting functions (circular bipartite, alluvial, barcodes, dual potentials) move to `fitkit.diagnostics.plots`.
- Comparison helpers (FC vs ECI scatterplots, correlation tables) move to `fitkit.diagnostics.comparison`.

**5. Notebook remains the narrative**
- After extraction, the notebook imports from `fitkit` and focuses on storytelling: motivation, choices, visualizations, and interpretation.
- Code cells become concise calls to package functions.

### Adapter Interface (REQ-0004)

The `DataLoader` interface is designed to be **Lynguine-compatible** but **locally implemented** for now:

```python
from dataclasses import dataclass
from typing import Any, Optional
import scipy.sparse as sp
import numpy as np

@dataclass
class DataBundle:
    """In-memory representation for bipartite incidence analysis."""
    matrix: sp.spmatrix          # Sparse incidence or count matrix
    row_labels: np.ndarray       # e.g. user IDs or country codes
    col_labels: np.ndarray       # e.g. words or product codes
    metadata: dict[str, Any]     # Provenance, cache info, thresholds, etc.

class DataLoader:
    """Abstract interface for data loading adapters."""
    def load(self) -> DataBundle:
        raise NotImplementedError
```

Future integration with Lynguine would provide alternative `DataLoader` implementations that delegate to `lynguine.access`, but the `DataBundle` contract remains stable.

### Authentication Strategy

**Current state**: The notebook uses `google.colab.auth.authenticate_user()` for BigQuery access, which only works in Colab.

**Design decision**: Authentication is a **data-loading concern**, not an algorithm concern. The `WikipediaLoader` will handle auth detection and delegation:

1. **In Colab**: Use `google.colab.auth.authenticate_user()` (current behavior).
2. **In local Jupyter/scripts**: Use Application Default Credentials (ADC) via `google.auth.default()`, which picks up credentials from:
   - `gcloud auth application-default login` (most common for local development)
   - Service account keys (via `GOOGLE_APPLICATION_CREDENTIALS` env var)
   - Compute Engine/Cloud Run service accounts (for cloud environments)
3. **In tests/CI**: Use synthetic fixtures only (no auth required, per REQ-0005).
4. **For advanced users**: Accept an optional pre-authenticated `bigquery.Client` in the loader constructor.

**Future Lynguine integration**: When data loading is delegated to Lynguine's access layer, Lynguine would own auth/credential management for all data sources (BigQuery, Cloud Storage, APIs). Fitkit's algorithms would remain auth-agnostic, receiving only in-memory `DataBundle` objects.

**Implementation notes**:
- The loader should gracefully degrade if auth fails, providing a clear error message pointing to setup docs.
- The notebook should document both Colab and local Jupyter workflows.
- ADC is the standard pattern for Google Cloud auth outside Colab and aligns with how Lynguine would handle credentials.

### Key Extracted Functions

From the notebook:

**Data loading** (`fitkit/data/loaders.py`):
- `QueryConfig` (dataclass for BigQuery parameters)
- `load_or_query_wikipedia(cfg, cache_path)` → wraps into `WikipediaLoader.load()`
- `generate_synthetic_data(cfg)` → wraps into `SyntheticLoader.load()`

**Algorithms** (`fitkit/algorithms/`):
- `fitness_complexity(M_bin, n_iter, tol)` → `fitkit/algorithms/fitness.py`
- `compute_eci_pci(M_bin)` → `fitkit/algorithms/eci.py`
- `sinkhorn_masked(M_bin, r, c, n_iter, tol, eps)` → `fitkit/algorithms/sinkhorn.py`

**Diagnostics** (`fitkit/diagnostics/`):
- `plot_circular_bipartite_flow(...)` → `fitkit/diagnostics/plots.py`
- `plot_alluvial_bipartite(...)` → `fitkit/diagnostics/plots.py`
- `plot_dual_potential_bipartite(...)` → `fitkit/diagnostics/plots.py`
- `plot_ranked_barcodes(...)` → `fitkit/diagnostics/plots.py`
- Helper to compute FC + ECI together → `fitkit/diagnostics/comparison.py`

### Testing Strategy (REQ-0002)

- Use `pytest` as the test runner.
- Create `tests/` directory with:
  - `test_fitness.py`: unit tests for Fitness-Complexity iteration on small fixtures.
  - `test_eci.py`: unit tests for ECI/PCI computation.
  - `test_sinkhorn.py`: unit tests for masked Sinkhorn scaling (convergence, marginal matching, edge cases).
  - `test_loaders.py`: tests for data-loading adapters using synthetic fixtures (no network/credentials required).
- Tests verify numerical properties (convergence, scale-free gauge, harmonic aggregation) and structural invariants (shapes, sparsity preservation).

### Non-Goals

- This CIP does **not** adopt Lynguine's full access/assess/address stack inside fitkit.
- This CIP does **not** change the notebook's scientific narrative or add new analyses.
- This CIP does **not** add a CLI or webapp (fitkit remains a library + notebook).

## Implementation Plan

**Phase 1: Minimal package scaffolding**
1. Create `fitkit/` directory structure and `__init__.py` files.
2. Add `pyproject.toml` with dependencies (numpy, scipy, pandas, scikit-learn, matplotlib, plotly).
3. Extract `types.py` with `DataBundle` and `DataLoader` protocol.

**Phase 2: Extract algorithms**
1. Move `fitness_complexity`, `compute_eci_pci`, `sinkhorn_masked` into `fitkit/algorithms/`.
2. Keep signatures unchanged; ensure they operate on in-memory inputs only.
3. Add docstrings with references to the paper.

**Phase 3: Extract data loaders**
1. Define `DataLoader` protocol and `DataBundle` in `fitkit/data/loaders.py`.
2. Implement `WikipediaLoader` (wraps the current `load_or_query_wikipedia` + `generate_synthetic_data`).
3. Implement `SyntheticLoader` for tests.
4. Move `QueryConfig` into `fitkit/data/loaders.py`.

**Phase 4: Extract diagnostics**
1. Move plotting functions into `fitkit/diagnostics/plots.py`.
2. Add a comparison helper in `fitkit/diagnostics/comparison.py` that computes FC + ECI/PCI together and returns aligned tables.

**Phase 5: Add tests**
1. Create `tests/` directory with initial test files.
2. Write unit tests for core algorithms using fixtures from `fitkit.data.fixtures`.
3. Ensure `pytest` runs offline by default (no network/cloud required).
4. Add `pytest.ini` or `pyproject.toml` test configuration.

**Phase 6: Update notebook**
1. Replace inline function definitions with imports from `fitkit`.
2. Simplify notebook cells to focus on narrative and interpretation.
3. Verify notebook runs end-to-end with package installed.
4. Update notebook's opening markdown to note that it uses the `fitkit` package.

## Backward Compatibility

The notebook currently has no external consumers (it's a standalone research artifact). Once the package is extracted, the notebook will depend on `fitkit`, but this is an internal reorganization with no backward compatibility concerns for external users.

## Testing Strategy

See Phase 5 above. Tests will use `pytest` and synthetic fixtures. No cloud credentials or network access required for default test runs.

Key test categories:
- **Numerical correctness**: Known fixtures with expected outputs (small matrices where FC/ECI/Sinkhorn can be computed by hand or validated against published results).
- **Convergence**: Ensure iterative methods converge within expected iterations.
- **Edge cases**: Isolated nodes, empty rows/columns, degenerate support.
- **Sparsity preservation**: Algorithms never densify sparse inputs.
- **Determinism**: All outputs are deterministic for fixed inputs and parameters.

## Related Requirements

This CIP implements solutions for:
- **REQ-0002**: Testing and verification (offline tests, deterministic fixtures, core algorithm coverage).
- **REQ-0003**: Algorithm/data separation (algorithms in `fitkit/algorithms/`, I/O in `fitkit/data/`).
- **REQ-0004**: Adapter interface (defines `DataLoader` protocol and `DataBundle` for future Lynguine integration).

## Implementation Status

- [ ] Create package structure and scaffolding
- [ ] Extract algorithm code into modules
- [ ] Extract data-loading code with adapter interface
- [ ] Extract diagnostic/plotting utilities
- [ ] Add test suite with fixtures
- [ ] Update notebook to import from package

## References

- Paper: `~/lawrennd/economic-fitness/economic-fitness.tex`
- Notebook: `wikipedia_editing_fitness_complexity.ipynb`
- Lynguine reference architecture: `~/lawrennd/lynguine/` (for DOA patterns)
- Related tenets: `explicit-boundaries-for-scientific-software`, `integration-friendly-data-loading`, `notebook-as-narrative-backed-by-functions`
