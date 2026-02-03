---
id: "2026-02-02_create-fitkit-package-structure"
title: "Create fitkit package structure and extract core algorithms/loaders"
status: "Completed"
priority: "High"
created: "2026-02-02"
last_updated: "2026-02-02"
category: "infrastructure"
related_cips:
  - "0002"
owner: "Neil Lawrennd"
dependencies: []
tags:
  - "backlog"
  - "package-structure"
  - "algorithms"
  - "data-loading"
---

# Task: Create fitkit package structure and extract core algorithms/loaders

> **Note**: This task implements CIP-0002 phases 1-3 (scaffolding, algorithms, data loaders).

## Description

Create the `fitkit/` Python package directory structure and extract core computational code from the notebook. This includes:
- Package scaffolding (`__init__.py`, `pyproject.toml`, type definitions)
- Algorithm modules (Fitness-Complexity, ECI/PCI, Sinkhorn)
- Data-loading adapter interface and implementations (Wikipedia/BigQuery, synthetic)

This establishes the architectural boundaries defined in REQ-0003 (algorithm/data separation) and REQ-0004 (adapter interface).

## Acceptance Criteria

- [ ] Package structure exists: `fitkit/`, `fitkit/algorithms/`, `fitkit/data/`, `fitkit/types.py`
- [ ] `pyproject.toml` defines package metadata and dependencies (numpy, scipy, pandas, scikit-learn, matplotlib, google-cloud-bigquery)
- [ ] `fitkit/types.py` defines `DataBundle` and `DataLoader` protocol
- [ ] `fitkit/algorithms/fitness.py` contains `fitness_complexity()` function (extracted from notebook)
- [ ] `fitkit/algorithms/eci.py` contains `compute_eci_pci()` function (extracted from notebook)
- [ ] `fitkit/algorithms/sinkhorn.py` contains `sinkhorn_masked()` function (extracted from notebook)
- [ ] `fitkit/data/loaders.py` defines `DataLoader` interface, `DataBundle`, and `QueryConfig`
- [ ] `fitkit/data/loaders.py` implements `WikipediaLoader` with environment-aware auth (Colab vs ADC)
- [ ] `fitkit/data/fixtures.py` implements `SyntheticLoader` for offline testing
- [ ] All algorithm functions accept in-memory inputs only (no I/O)
- [ ] All extracted functions have docstrings referencing the paper

## Implementation Notes

**Package structure**:
```
fitkit/
├── __init__.py
├── types.py
├── algorithms/
│   ├── __init__.py
│   ├── fitness.py
│   ├── eci.py
│   └── sinkhorn.py
└── data/
    ├── __init__.py
    ├── loaders.py
    └── fixtures.py
```

**Auth strategy** (per CIP-0002):
- Detect environment: check for `google.colab` module
- In Colab: use `google.colab.auth.authenticate_user()`
- Outside Colab: use `google.auth.default()` (ADC)
- Provide clear error messages if auth fails

**Extraction approach**:
- Copy function definitions from notebook cells
- Keep signatures unchanged
- Add type hints where possible
- Update docstrings to be standalone (not notebook-dependent)

## Related

- CIP: 0002
- Requirements: REQ-0003 (separation), REQ-0004 (adapter interface)

## Progress Updates

### 2026-02-02

Task created to implement CIP-0002 phases 1-3 (package structure, algorithms, data loaders).

Task completed:
- Created package structure: fitkit/, fitkit/algorithms/, fitkit/data/
- Created pyproject.toml with dependencies and tool configurations
- Created fitkit/types.py with DataBundle and DataLoader protocol
- Extracted fitness_complexity() into fitkit/algorithms/fitness.py
- Extracted compute_eci_pci() into fitkit/algorithms/eci.py  
- Extracted sinkhorn_masked() into fitkit/algorithms/sinkhorn.py
- Created WikipediaLoader with environment-aware auth (Colab vs ADC) in fitkit/data/loaders.py
- Created SyntheticLoader and create_small_fixture() in fitkit/data/fixtures.py
- All __init__.py files created with proper exports
- All functions have comprehensive docstrings referencing the paper
