---
author: "Neil Lawrence"
created: "2026-02-10"
id: "000B"
last_updated: "2026-02-10"
status: "Implemented"
compressed: false
related_requirements: []
related_cips: []
tags:
- cip
- data-access
- atlas
- datasets
title: "Atlas of Economic Complexity Data Access"
---

# CIP-000B: Atlas of Economic Complexity Data Access

> **Note**: CIPs describe HOW to achieve requirements (WHAT).  
> Use `related_requirements` to link to the requirements this CIP implements.

> **Compression Metadata**: The `compressed` field tracks whether this CIP's key decisions have been compressed into formal documentation (README, Sphinx, architecture docs). Set to `false` by default. After closing a CIP and updating formal documentation with its essential outcomes, set `compressed: true`. This enables `whats-next` to prompt for documentation compression.

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented (add reason, use superseded_by if replaced)
- [ ] Deferred - Postponed (use blocked_by field to indicate blocker)

## Summary

Implement dataset loaders for the Harvard Growth Lab's Atlas of Economic Complexity, providing easy access to international trade data spanning 1962-2024 with multiple classification systems (HS92, SITC). This enables fitkit users to analyze real-world economic data without manual data wrangling.

## Motivation

The current `fitkit.datasets` module only provides a single dataset (world trade 1998-2000) extracted from the R economiccomplexity package. To enable comprehensive economic fitness analysis, we need access to:

1. **Multiple time periods**: Compare economic evolution across decades
2. **Recent data**: Analysis of current trade patterns (through 2024)
3. **Historical data**: Long-term trends (back to 1962 with SITC)
4. **Flexible aggregation**: Different product classification levels (2/4/6-digit)

The Harvard Growth Lab's Atlas of Economic Complexity is the gold standard dataset for this domain, widely used in research and policy. Providing seamless access to Atlas data through fitkit will:
- Enable time-series fitness analysis
- Support comparative studies across years
- Facilitate replication of published research
- Lower the barrier to entry for economic complexity analysis

## Detailed Description

### Architecture

The implementation follows the existing `fitkit.datasets` module pattern with two new public functions:

1. **`load_atlas_trade(year, classification, product_level, rca_threshold, auto_download)`**
   - Downloads and caches raw trade data from Harvard Dataverse
   - Computes Balassa Index (Revealed Comparative Advantage)
   - Returns sparse binary matrix + metadata DataFrames
   - Consistent API with existing `load_world_trade_1998_2000()`

2. **`list_atlas_available_years(classification, auto_download)`**
   - Helper to discover data coverage
   - Returns list of available years for a classification

3. **`load_atlas_product_names(classification)`**
   - Loads product classification table with human-readable names
   - Maps product codes (e.g., '0101', '8703') to descriptions (e.g., 'Horses', 'Cars')
   - Includes hierarchical structure (sections, chapters, products)
   - Small files (~80KB for HS92, ~20KB for SITC) downloaded from Atlas S3
   - Enables interpretable analysis and visualization

### Data Source: Harvard Dataverse API

Atlas data is hosted on Harvard Dataverse at:
- Dataset DOI: `doi:10.7910/DVN/T4CHWJ` (HS92)
- Dataset DOI: `doi:10.7910/DVN/H8SFD2` (SITC)

**Key technical challenges:**
1. **Dataverse API redirects**: The direct file access API (`/api/access/datafile/{id}`) returns a 303 redirect to a time-limited signed S3 URL
2. **Large file sizes**: Country-product-year files are 50-475 MB
3. **Column name variations**: Dataset schema varies slightly across versions
4. **Authentication**: Some Dataverse endpoints require user-agent headers

**Solution approach:**
- Use `urllib.request.urlopen()` which follows redirects automatically
- Add proper user-agent headers: `User-Agent: fitkit/1.0 (Python)`
- Implement progress indication for large downloads
- Local caching in `fitkit/data/atlas/` to avoid re-downloading
- Flexible column detection to handle schema variations

### RCA Computation

The Revealed Comparative Advantage (Balassa Index) is computed as:

```
RCA_cp = (X_cp / X_c) / (X_p / X_total)
```

Where:
- `X_cp` = exports of product p from country c
- `X_c` = total exports from country c
- `X_p` = total global exports of product p
- `X_total` = total global trade

Binary matrix: `M[c,p] = 1` if `RCA_cp >= threshold` (default: 1.0)

### Product Aggregation

HS92 products are 6-digit codes (e.g., "851712"). Users can aggregate to:
- **2-digit**: Very broad categories (~20-40 products)
- **4-digit**: Standard level (recommended, ~90-200 products)
- **6-digit**: Most detailed (thousands of products, very sparse)

Aggregation is done by truncating codes and summing export values.

### Caching Strategy

Downloaded data is stored in `fitkit/data/atlas/`:
```
fitkit/data/atlas/
├── atlas_hs92_country_product_year.csv (475 MB)
└── atlas_sitc_country_product_year.csv (~200 MB)
```

This directory is added to `.gitignore` to avoid committing large data files to version control.

## Implementation Plan

1. **Core Data Loading** ✅
   - Implement `_download_atlas_file()` helper with redirect handling
   - Add progress indication for large downloads
   - Implement file caching logic

2. **RCA Computation** ✅
   - Implement `_compute_balassa_index()` helper
   - Handle country-product aggregation
   - Create sparse matrix from results

3. **Public API** ✅
   - Implement `load_atlas_trade()` with comprehensive docstring
   - Implement `list_atlas_available_years()` helper
   - Add to `fitkit/__init__.py` exports
   - Support HS92 and SITC classifications

4. **Data Management** ✅
   - Add `fitkit/data/atlas/` to `.gitignore`
   - Handle missing data gracefully
   - Provide clear error messages

5. **Documentation** ✅
   - Comprehensive docstrings with usage examples
   - Create demonstration notebook `atlas_fitness_comparison.ipynb`
   - Show 2000 vs 2020 comparison analysis

6. **Testing** 🔄
   - Validate data loading and RCA computation
   - Test with multiple years and classifications
   - Verify matrix dimensions and sparsity

## Backward Compatibility

This is a purely additive change - no existing functionality is modified. New imports:
```python
from fitkit import load_atlas_trade, list_atlas_available_years
```

Existing code using `load_world_trade_1998_2000()` is unaffected.

## Testing Strategy

### Manual Testing (Completed)
- Downloaded HS92 dataset (475 MB) successfully
- Verified 35 years of data available (1988-2024)
- Loaded 2010 data: 231 countries × 98 products at 4-digit level
- Confirmed matrix density (~19%) is reasonable
- Verified RCA computation produces sensible results

### Automated Testing (Needed)
- Unit tests for `_compute_balassa_index()`
- Integration test with mock Dataverse responses
- Test product aggregation at different levels
- Verify error handling for missing years

## Example Usage

From the comprehensive demonstration notebook:

```python
from fitkit import load_atlas_trade, list_atlas_available_years
from fitkit.algorithms import fitness_complexity

# Check available years
years = list_atlas_available_years('hs92')
print(f"Data available: {years[0]}-{years[-1]}")

# Load data for analysis
M_2000, countries_2000, products_2000 = load_atlas_trade(
    year=2000, 
    classification='hs92',
    product_level=4
)

# Compute fitness
F, Q, history = fitness_complexity(M_2000)

# Analyze results
countries_2000['fitness'] = F
print(countries_2000.nlargest(10, 'fitness'))
```

## Implementation Status

- [x] Design data loader API
- [x] Implement Dataverse download with redirect handling
- [x] Implement RCA computation
- [x] Add product aggregation logic
- [x] Create comprehensive demonstration notebook
- [x] Add exports to package `__init__.py`
- [x] Update `.gitignore` for cached data
- [ ] Add unit tests
- [ ] Run test suite
- [ ] Update formal documentation (after validation)

## Files Modified

- `fitkit/datasets.py`: Added 285 lines with Atlas loaders
- `fitkit/__init__.py`: Exported new functions
- `.gitignore`: Added `fitkit/data/atlas/` exclusion
- `examples/atlas_fitness_comparison.ipynb`: Comprehensive demonstration

## Data Specifications

### HS92 (Harmonized System 1992)
- **Coverage**: 1988-2024 (35 years, most complete from 1995)
- **Products**: ~5,000 at 6-digit level
- **File size**: 475 MB
- **Dataverse ID**: 13439447 (4-digit aggregated)

### SITC (Standard International Trade Classification)
- **Coverage**: 1962-2023 (62 years)
- **Products**: ~700 at 4-digit level
- **File size**: ~200 MB
- **Dataverse ID**: TBD

## References

- Harvard Growth Lab Atlas: https://atlas.hks.harvard.edu/
- Dataverse dataset: https://dataverse.harvard.edu/dataverse/atlas
- Hausmann, R., et al. (2014). *The Atlas of Economic Complexity*. MIT Press.
- Data citation: The Growth Lab at Harvard University, 2024, "International Trade Data (HS92), 1988-2024", https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/T4CHWJ, Harvard Dataverse
