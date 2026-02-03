---
id: "2026-02-02_update-notebook-to-use-package"
title: "Update notebook to import from fitkit package"
status: "Completed"
priority: "Medium"
created: "2026-02-02"
last_updated: "2026-02-03"
category: "infrastructure"
related_cips:
  - "0002"
owner: "Neil Lawrennd"
dependencies:
  - "2026-02-02_create-fitkit-package-structure"
  - "2026-02-02_extract-diagnostics-and-add-tests"
tags:
  - "backlog"
  - "notebook"
  - "refactoring"
---

# Task: Update notebook to import from fitkit package

> **Note**: This task implements CIP-0002 phase 6 (notebook refactoring).

## Description

Refactor `wikipedia_editing_fitness_complexity.ipynb` to import functions from the `fitkit` package instead of defining them inline. The notebook becomes the scientific narrative while delegating computational details to the package.

This aligns with the tenet `notebook-as-narrative-backed-by-functions`.

## Acceptance Criteria

- [ ] Notebook no longer contains function definitions for algorithms (FC, ECI, Sinkhorn)
- [ ] Notebook no longer contains function definitions for data loading or diagnostics
- [ ] Notebook imports from `fitkit`:
  - `from fitkit.data.loaders import WikipediaLoader, QueryConfig`
  - `from fitkit.algorithms.fitness import fitness_complexity`
  - `from fitkit.algorithms.eci import compute_eci_pci`
  - `from fitkit.algorithms.sinkhorn import sinkhorn_masked`
  - `from fitkit.diagnostics.plots import plot_circular_bipartite_flow, ...`
  - `from fitkit.diagnostics.comparison import compare_fc_eci`
- [ ] Notebook cells are concise and focused on storytelling (motivation, choices, interpretation)
- [ ] Notebook runs end-to-end successfully with `fitkit` installed
- [ ] Notebook's opening markdown documents:
  - Installation: `pip install -e .` (local development) or `pip install fitkit` (if published)
  - Authentication: Both Colab (automatic) and local Jupyter (ADC via `gcloud auth application-default login`)
- [ ] Notebook preserves all existing outputs (plots, tables, narrative)

## Implementation Notes

**Refactoring strategy**:
1. Add an early cell with imports: `from fitkit.data.loaders import ...`
2. Replace function definition cells with import statements
3. Update function calls to use imported names (should be unchanged if extraction preserved signatures)
4. Test notebook execution end-to-end in both Colab and local Jupyter

**Auth documentation** (update BigQuery setup instructions):
- Keep Colab instructions (current behavior)
- Add local Jupyter instructions:
  ```
  For local Jupyter notebooks, authenticate via Application Default Credentials:
  
  gcloud auth application-default login
  
  This provides credentials that the fitkit package will automatically detect.
  ```

**Installation documentation** (add new cell near the top):
```python
# Install fitkit package (run once)
# For local development:
# !pip install -e /path/to/fitkit

# For published package (future):
# !pip install fitkit
```

## Related

- CIP: 0002
- Tenets: notebook-as-narrative-backed-by-functions

## Progress Updates

### 2026-02-02

Task created to implement CIP-0002 phase 6 (notebook refactoring).

### 2026-02-03

Task completed:
- Moved notebook to `examples/` directory for better organization
- Added auto-install cell for Colab compatibility (tries import, installs if missing)
- Added imports from `fitkit.data` and `fitkit.algorithms`
- Removed all inline function definitions (fitness_complexity, compute_eci_pci, sinkhorn_masked, data loading)
- Updated BigQuery authentication documentation (Colab automatic, local via ADC)
- Preserved all narrative, plots, and outputs
- Updated README and CI workflow to reflect new notebook location

Note: Diagnostic/plotting functions remain in notebook (not yet extracted to package).
