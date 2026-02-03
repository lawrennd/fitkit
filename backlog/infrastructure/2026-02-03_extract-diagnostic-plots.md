---
id: "2026-02-03_extract-diagnostic-plots"
title: "Extract diagnostic plotting functions to fitkit.diagnostics"
status: "Completed"
priority: "Medium"
created: "2026-02-03"
last_updated: "2026-02-03"
category: "infrastructure"
related_cips:
  - "0002"
owner: "Neil Lawrennd"
dependencies:
  - "2026-02-02_create-fitkit-package-structure"
tags:
  - "backlog"
  - "diagnostics"
  - "visualisation"
---

# Task: Extract diagnostic plotting functions to fitkit.diagnostics

> **Note**: This completes the deferred portion of CIP-0002 phase 4 (diagnostics extraction).

## Description

Extract the 4 visualization functions currently in the notebook to a `fitkit/diagnostics/plots.py` module. These functions provide flow-native visualizations for bipartite data analysis and should be reusable across different notebooks and scripts.

## Acceptance Criteria

- [ ] `fitkit/diagnostics/` directory created
- [ ] `fitkit/diagnostics/__init__.py` exports plotting functions
- [ ] `fitkit/diagnostics/plots.py` contains extracted plotting functions:
  - `plot_circular_bipartite_flow()` - Circular layout with flow edges
  - `plot_alluvial_bipartite()` - Alluvial/Sankey-style bipartite flow
  - `plot_dual_potential_bipartite()` - Dual potential (fitness/complexity) visualization
  - `plot_ranked_barcodes()` - Ranked barcode comparison plots
- [ ] Functions maintain existing signatures (backward compatible)
- [ ] Functions are documented with docstrings (parameters, returns, examples)
- [ ] Notebook updated to import from `fitkit.diagnostics` instead of defining inline
- [ ] All existing plot outputs in notebook are preserved (visual regression test)
- [ ] CI passes with updated imports

## Implementation Notes

**Extraction strategy**:
1. Copy plotting function definitions from notebook cell to `fitkit/diagnostics/plots.py`
2. Add proper module docstring and function docstrings
3. Add type hints for parameters
4. Update notebook to: `from fitkit.diagnostics import plot_circular_bipartite_flow, ...`
5. Remove function definition cells from notebook
6. Verify all plots still render correctly

**Dependencies**:
- matplotlib (already in dependencies)
- No additional dependencies needed

**Testing**:
- Visual: Run notebook end-to-end and verify all plots render
- Unit tests optional (plotting functions are primarily visual)

## Related

- CIP: 0002 (phase 4 - diagnostics)
- Related task: 2026-02-02_extract-diagnostics-and-add-tests (test portion completed, diagnostics deferred)

## Progress Updates

### 2026-02-03

Task created to complete the diagnostics extraction portion of CIP-0002 phase 4.

Task completed:
- Created `fitkit/diagnostics/` module with `__init__.py` and `plots.py`
- Extracted 4 plotting functions with proper docstrings
- Updated notebook to import from `fitkit.diagnostics`
- Removed inline function definitions from notebook
- All plot outputs preserved
