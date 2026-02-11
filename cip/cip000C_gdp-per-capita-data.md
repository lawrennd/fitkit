---
author: "Neil Lawrence"
created: "2026-02-11"
id: "000C"
last_updated: "2026-02-11"
status: "Proposed"
compressed: false
related_requirements: []
related_cips: ["000B"]  # Atlas data access established the pattern
tags:
- cip
- data-access
- gdp
- world-bank
- validation
title: "World Bank Indicators Data Access"
---

# CIP-000C: World Bank Indicators Data Access

> **Note**: CIPs describe HOW to achieve requirements (WHAT).  
> Use `related_requirements` to link to the requirements this CIP implements.

> **Compression Metadata**: The `compressed` field tracks whether this CIP's key decisions have been compressed into formal documentation (README, Sphinx, architecture docs). Set to `false` by default. After closing a CIP and updating formal documentation with its essential outcomes, set `compressed: true`.

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Implement flexible dataset loaders for World Bank indicators (GDP per capita, Human Capital Index, etc.) from the World Development Indicators (WDI) database. This enables validation of economic fitness scores against multiple dimensions of economic development and human capital, complementing the trade data from CIP-000B.

## Motivation

Economic fitness scores are theoretical constructs derived from trade patterns. To validate these scores and enable multidimensional economic analysis, we need access to real economic and social outcomes beyond just trade data.

The World Bank's World Development Indicators (WDI) provides authoritative data across multiple dimensions:

### GDP Per Capita (Primary Use Case)
GDP per capita is the gold standard metric for economic development:
1. **Validate fitness scores**: Check if fitness correlates with actual economic performance
2. **Time-series analysis**: Track how fitness relates to economic growth
3. **Country comparisons**: Compare fitness rankings with traditional economic rankings
4. **Research replication**: Enable reproduction of published fitness-GDP studies

### Human Capital Index (Secondary Use Case)
The Human Capital Index (HCI) measures the human capital that a child born today can expect to attain by age 18:
1. **Education-fitness relationship**: Analyze how human capital investment relates to economic complexity
2. **Development pathways**: Compare countries at similar fitness levels but different HCI scores
3. **Policy insights**: Identify whether complexity precedes or follows human capital development
4. **Productivity analysis**: HCI captures health and education components of productivity

### Other Indicators (Future Extensions)
The flexible design enables easy addition of other indicators:
- **Trade metrics**: Exports/imports as % of GDP, trade openness
- **Innovation**: R&D expenditure, patent applications, high-tech exports
- **Infrastructure**: Internet penetration, electricity access, logistics performance
- **Inequality**: Gini coefficient, income shares
- **Demographics**: Population, urbanization, labor force participation

### Why World Bank WDI?
- Comprehensive coverage (200+ countries, 1960-present)
- Regularly updated (annual releases)
- Standardized methodology across indicators
- Free public access via API
- Widely used in academic research
- Single API interface for all indicators

## Detailed Description

### Architecture

Following the pattern established in CIP-000B, implement flexible dataset loaders in `fitkit.datasets`:

#### Core Function
**`load_worldbank_indicator(indicator_code, countries=None, start_year=None, end_year=None, auto_download=True)`**
- General-purpose loader for any World Bank indicator
- Takes indicator code (e.g., 'NY.GDP.PCAP.CD', 'HD.HCI.OVRL')
- Downloads data from World Bank API
- Caches locally per indicator for reuse
- Returns pandas DataFrame with years as columns, countries as rows
- Optional filtering by country list and date range

#### Convenience Functions
**`load_gdp_per_capita(countries=None, start_year=None, end_year=None, auto_download=True)`**
- Wrapper for `load_worldbank_indicator('NY.GDP.PCAP.CD', ...)`
- More discoverable API for common use case
- Same parameters as core function

**`load_human_capital_index(countries=None, start_year=None, end_year=None, auto_download=True)`**
- Wrapper for `load_worldbank_indicator('HD.HCI.OVRL', ...)`
- Enables analysis of human capital vs economic complexity
- Same parameters as core function

#### Helper Functions
**`list_worldbank_indicators(search_term=None)`**
- Browse available World Bank indicators
- Optional search by keyword (e.g., "education", "health", "trade")
- Returns DataFrame with indicator codes, names, and descriptions

**`list_worldbank_available_countries(indicator_code, auto_download=True)`**
- Helper to discover data coverage for specific indicator
- Returns list of country codes with available data

### Data Source: World Bank API

World Bank provides free programmatic access through their Indicators API:
- **Base endpoint**: `https://api.worldbank.org/v2/country/all/indicator/{INDICATOR_CODE}`
- **Format**: JSON or CSV
- **No authentication required**

#### Key Indicator Codes

**GDP Per Capita (Primary)**
- `NY.GDP.PCAP.CD` - GDP per capita, current US$
- `NY.GDP.PCAP.PP.CD` - GDP per capita, PPP (better for cross-country comparison)
- `NY.GDP.PCAP.KD` - GDP per capita, constant 2015 US$ (better for time-series)

**Human Capital Index (Secondary)**
- `HD.HCI.OVRL` - Human Capital Index (overall score, 0-1 scale)
- `HD.HCI.EYRS` - Expected years of school
- `HD.HCI.LAYS` - Learning-adjusted years of school
- `HD.HCI.HLOS` - Harmonized test scores
- `HD.HCI.MORT` - Survival rate to age 5

**Other Useful Indicators** (for future use)
- `NE.TRD.GNFS.ZS` - Trade (% of GDP)
- `GB.XPD.RSDV.GD.ZS` - R&D expenditure (% of GDP)
- `SP.POP.TOTL` - Total population
- `SI.POV.GINI` - Gini index (inequality)

**API Query Parameters:**
- `format=json` - JSON output
- `date=2000:2024` - Date range filter
- `per_page=20000` - Results per page (avoid pagination)
- `page=1` - Page number

**Example URL:**
```
https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json&date=1960:2024&per_page=20000
```

**Response Structure:**
```json
[
  {"page": 1, "pages": 1, "per_page": 20000, "total": 13000},
  [
    {
      "indicator": {"id": "NY.GDP.PCAP.CD", "value": "GDP per capita (current US$)"},
      "country": {"id": "US", "value": "United States"},
      "countryiso3code": "USA",
      "date": "2023",
      "value": 76398.59,
      "unit": "",
      "obs_status": "",
      "decimal": 2
    },
    ...
  ]
]
```

### Implementation Strategy

**Key design decisions:**

1. **Unified core function** with indicator-specific wrappers
   - Core: `load_worldbank_indicator(indicator_code, ...)`
   - Wrappers: `load_gdp_per_capita()`, `load_human_capital_index()`
   - Easy to add new indicators without duplicating download logic

2. **Direct API calls** (no external dependencies)
   - Use `urllib.request` for consistency with CIP-000B
   - Parse JSON response with `json` module
   - Convert to pandas DataFrame
   - Same pattern as `_download_atlas_file()` from CIP-000B

3. **Per-indicator caching** in `fitkit/data/worldbank/`
   - Cache each indicator separately: `gdp_per_capita.csv`, `human_capital_index.csv`
   - Add directory to `.gitignore`
   - Check cache before API call
   - Cache file naming: `{indicator_code.replace('.', '_')}.csv`

4. **Country code normalization**
   - World Bank uses ISO3 codes (USA, GBR, CHN)
   - Match Atlas data format for easy joining
   - Provide mapping function for ISO2 ↔ ISO3 (future enhancement)

5. **Missing data handling**
   - WDI data has gaps (conflicts, missing reports, new indicators)
   - Return `NaN` for missing values
   - Document coverage in docstring per indicator
   - HCI only available from 2010 onwards (vs GDP from 1960)

6. **Indicator metadata**
   - Store indicator name and description in cache
   - Add as DataFrame attrs for reference
   - Useful for plot labels and documentation

### Data Coverage

#### GDP Per Capita
**Time coverage**: 1960-present (typically 1-2 years lag)

**Geographic coverage**: 200+ countries and territories
- All UN member states
- Some territories and regions
- Regional/income group aggregates (exclude in analysis)

**Known limitations:**
- Missing data for small territories
- Gaps during conflicts/political transitions
- Recent years may have provisional estimates
- Some historical data revised retrospectively

#### Human Capital Index
**Time coverage**: 2010, 2012, 2017, 2020 (released periodically, not annual)

**Geographic coverage**: 170+ countries
- Most countries with sufficient education/health data
- Smaller coverage than GDP (requires detailed education/health statistics)
- No data for small territories or conflict zones

**Known limitations:**
- Not available for all years (periodic releases)
- Missing for countries with poor education/health data systems
- Methodology changes between releases (use with caution for trends)
- Most recent data preferred for cross-sectional analysis

#### General Indicator Differences
Different indicators have different:
- Time coverage (GDP: 1960+, HCI: 2010+, others vary)
- Geographic coverage (some indicators only for certain income groups)
- Update frequency (GDP: annual, HCI: periodic, others vary)
- Data quality (developed countries generally more complete)

### Integration with Fitness Analysis

Common analysis patterns enabled by this data:

```python
# Example 1: Fitness-GDP correlation
M, countries, products = load_atlas_trade(year=2020)
F = fitness_complexity(M)[0]
gdp_df = load_gdp_per_capita(start_year=2020, end_year=2020)

merged = countries.merge(
    gdp_df[['2020']], 
    left_on='location_code', 
    right_index=True
)
merged['fitness'] = F
correlation = merged[['fitness', '2020']].corr()

# Example 2: Fitness-GDP-HCI triangle
hci_df = load_human_capital_index(start_year=2020, end_year=2020)
analysis = countries.copy()
analysis['fitness'] = F
analysis = analysis.merge(gdp_df[['2020']], left_on='location_code', right_index=True)
analysis = analysis.merge(hci_df[['2020']], left_on='location_code', right_index=True, suffixes=('_gdp', '_hci'))
# Analyze: Does fitness predict GDP given HCI? Does HCI mediate fitness→GDP?

# Example 3: Fitness changes vs GDP growth
gdp_2000_2020 = load_gdp_per_capita(start_year=2000, end_year=2020)
gdp_growth = (gdp_2000_2020['2020'] - gdp_2000_2020['2000']) / gdp_2000_2020['2000']
# Compare with fitness changes over same period

# Example 4: Custom indicator
trade_openness = load_worldbank_indicator('NE.TRD.GNFS.ZS', start_year=2020, end_year=2020)
# Analyze relationship between trade openness and complexity
```

## Implementation Plan

1. **Core API Access** ⏳
   - Implement `_download_worldbank_indicator()` helper
   - Handle JSON parsing and error cases
   - Add progress indication for large downloads
   - Extract indicator metadata from response

2. **Data Processing** ⏳
   - Convert JSON to pandas DataFrame
   - Pivot to years-as-columns format
   - Handle missing values appropriately
   - Store indicator metadata in DataFrame attrs

3. **Core Public API** ⏳
   - Implement `load_worldbank_indicator(indicator_code, ...)` with comprehensive docstring
   - Implement `list_worldbank_indicators(search_term)` for browsing
   - Implement `list_worldbank_available_countries(indicator_code)` helper
   - Support country and date filtering

4. **Convenience Wrappers** ⏳
   - Implement `load_gdp_per_capita()` wrapper
   - Implement `load_human_capital_index()` wrapper
   - Add to `fitkit/__init__.py` exports
   - Consider adding more wrappers for common indicators

5. **Data Management** ⏳
   - Add `fitkit/data/worldbank/` to `.gitignore`
   - Implement per-indicator caching logic
   - Provide clear error messages for missing indicators
   - Handle indicator-specific quirks (HCI periodic vs GDP annual)

6. **Documentation** ⏳
   - Comprehensive docstrings with usage examples
   - Update `atlas_fitness_comparison.ipynb` to include GDP and HCI validation
   - Show correlation analysis and triangle plots (Fitness-GDP-HCI)
   - Document common indicator codes

7. **Testing** ⏳
   - Validate data loading and parsing for multiple indicators
   - Test with different country/year filters
   - Verify DataFrame structure consistency
   - Test merging with fitness data
   - Test HCI periodic data handling

## Backward Compatibility

This is a purely additive change. New imports:
```python
from fitkit import load_gdp_per_capita, list_gdp_available_countries
```

No existing functionality is modified.

## Testing Strategy

### Manual Testing (Needed)
- Download full dataset and verify structure
- Check specific countries (USA, CHN, GBR)
- Verify missing data handling
- Test date range filtering
- Confirm merge with Atlas data

### Automated Testing (Needed)
- Unit tests for JSON parsing
- Integration test with mock World Bank responses
- Test country filtering
- Test date range filtering
- Verify error handling

## Design Rationale

### Why Unified Core Function?

**Original idea**: Create separate `load_gdp_per_capita()` and `load_human_capital_index()` functions.

**Better approach**: Unified `load_worldbank_indicator(code, ...)` with convenience wrappers.

**Advantages**:
1. **Single download/caching implementation**: All indicators share the same API interaction logic
2. **Easy extensibility**: Users can access any World Bank indicator without waiting for us to add wrappers
3. **Consistent behavior**: Same parameters, same DataFrame format, same caching strategy
4. **Less code duplication**: ~200 lines once vs ~200 lines per indicator
5. **Discoverable**: `list_worldbank_indicators('education')` enables exploration

**Convenience wrappers still valuable**:
- More discoverable than remembering indicator codes
- Better for autocomplete/IDE suggestions
- Can add indicator-specific documentation
- Common use cases remain simple: `load_gdp_per_capita()` vs `load_worldbank_indicator('NY.GDP.PCAP.CD')`

## Alternative Approaches Considered

### 1. Use `wbdata` or `wbgapi` Python packages
**Pros**: Convenient wrapper around World Bank API  
**Cons**: 
- Adds external dependency (requires `pip install`)
- Less control over caching strategy
- Different API patterns than CIP-000B (consistency matters)
- Extra dependency for relatively simple API

**Decision**: Use direct API calls for consistency and minimal dependencies

### 2. Use Penn World Table (PWT)
**Pros**: 
- More comprehensive economic data
- Research-quality adjustments
- GDP in multiple formats (PPP, constant prices)

**Cons**:
- More complex download process
- Less frequent updates (2-3 year lag)
- Requires understanding of PWT methodology
- Different data structure (harder to merge with WDI)

**Decision**: Start with World Bank for simplicity, consider PWT in future CIP if needed

### 3. Include World Bank or pre-download static file
**Pros**: No runtime API calls, faster load time  
**Cons**: 
- Outdated data in package
- Large package size (~50MB+ for all indicators)
- Violates "live data" principle
- Version management complexity

**Decision**: Use API with local caching (CIP-000B pattern)

### 4. Separate function per indicator
**Pros**: Simple to understand, explicit API  
**Cons**:
- Code duplication (download logic repeated)
- Hard to extend (need new function for each indicator)
- User can't access indicators we haven't wrapped
- More maintenance burden

**Decision**: Unified core with convenience wrappers (best of both worlds)

## Data Specifications

### GDP Per Capita (Current US$)
- **Indicator**: NY.GDP.PCAP.CD
- **Definition**: Gross domestic product divided by midyear population
- **Units**: Current US dollars
- **Source**: World Bank national accounts data, OECD National Accounts
- **Update frequency**: Annual
- **Coverage**: 1960-present (typically 1-2 years lag)

### Alternative: GDP Per Capita (PPP)
- **Indicator**: NY.GDP.PCAP.PP.CD
- **Definition**: GDP per capita based on purchasing power parity
- **Units**: Current international dollars
- **Advantage**: Better for cross-country comparisons (adjusts for price levels)
- **Status**: Can load via `load_worldbank_indicator('NY.GDP.PCAP.PP.CD', ...)`

### Human Capital Index (HCI)
- **Indicator**: HD.HCI.OVRL
- **Definition**: Amount of human capital that a child born today can expect to attain by age 18, given the risks of poor health and poor education
- **Scale**: 0 to 1 (1 = best possible human capital outcomes)
- **Components**: 
  - Survival (probability of survival to age 5)
  - Expected years of school
  - Harmonized test scores (quality of learning)
  - Adult survival rate (health environment)
- **Source**: World Bank Human Capital Project
- **Update frequency**: Periodic (2010, 2012, 2017, 2020, ...)
- **Coverage**: 170+ countries
- **Interpretation**: 
  - HCI = 0.50 means a child born today will be only half as productive as they could be with complete education and full health
  - Measures potential productivity, not just years of schooling
- **Reference**: https://www.worldbank.org/en/publication/human-capital

### Other Useful Indicators
Available via `load_worldbank_indicator(code, ...)`:
- **Trade**: `NE.TRD.GNFS.ZS` - Trade (% of GDP)
- **R&D**: `GB.XPD.RSDV.GD.ZS` - R&D expenditure (% of GDP)
- **Innovation**: `IP.PAT.RESD` - Patent applications by residents
- **Education**: `SE.TER.ENRR` - Tertiary school enrollment (%)
- **Health**: `SP.DYN.LE00.IN` - Life expectancy at birth
- **Infrastructure**: `IT.NET.USER.ZS` - Internet users (% of population)
- **Inequality**: `SI.POV.GINI` - Gini index

## Example Usage

```python
from fitkit import (
    load_gdp_per_capita, 
    load_human_capital_index,
    load_worldbank_indicator,
    load_atlas_trade, 
    fitness_complexity
)

# Example 1: GDP Per Capita Analysis
gdp_df = load_gdp_per_capita(start_year=2000, end_year=2020)
print(f"GDP data: {gdp_df.shape}")  # (200+ countries, 21 years)

# Merge with fitness data
M_2020, countries_2020, _ = load_atlas_trade(year=2020)
F_2020, _, _ = fitness_complexity(M_2020)
countries_2020['fitness'] = F_2020

# Join GDP and fitness
comparison = countries_2020.merge(
    gdp_df[['2020']], 
    left_on='location_code', 
    right_index=True,
    how='inner'
)
comparison = comparison.rename(columns={'2020': 'gdp_per_capita'})

# Analyze correlation
correlation = comparison[['fitness', 'gdp_per_capita']].corr().iloc[0, 1]
print(f"Fitness-GDP correlation: {correlation:.3f}")

# Example 2: Human Capital Index Analysis
hci_df = load_human_capital_index(start_year=2020, end_year=2020)
comparison_hci = comparison.merge(
    hci_df[['2020']],
    left_on='location_code',
    right_index=True,
    how='inner'
)
comparison_hci = comparison_hci.rename(columns={'2020': 'human_capital_index'})

# Analyze Fitness-GDP-HCI relationships
print("\nCorrelation Matrix:")
print(comparison_hci[['fitness', 'gdp_per_capita', 'human_capital_index']].corr())

# Countries with high fitness but low HCI (complexity before human capital?)
comparison_hci['fitness_rank'] = comparison_hci['fitness'].rank(ascending=False)
comparison_hci['hci_rank'] = comparison_hci['human_capital_index'].rank(ascending=False)
comparison_hci['rank_diff'] = comparison_hci['fitness_rank'] - comparison_hci['hci_rank']
print("\nHigh fitness, low HCI (negative rank_diff):")
print(comparison_hci.nsmallest(10, 'rank_diff')[['location_code', 'fitness_rank', 'hci_rank', 'rank_diff']])

# Example 3: Custom indicator (Trade Openness)
trade_df = load_worldbank_indicator('NE.TRD.GNFS.ZS', start_year=2020, end_year=2020)
print(f"\nTrade openness (% of GDP) available for {trade_df.shape[0]} countries")
```

## Implementation Status

### Phase 1: Core Infrastructure ✅
- [x] Design unified API interface (core + convenience wrappers)
- [x] Implement `_download_worldbank_indicator()` helper
- [x] Implement JSON parsing and DataFrame conversion
- [x] Implement per-indicator caching
- [x] Handle duplicate country-year entries

### Phase 2: Public Functions ✅
- [x] Implement `load_worldbank_indicator()` (core function)
- [x] Implement `load_gdp_per_capita()` (wrapper)
- [x] Implement `load_human_capital_index()` (wrapper)
- [x] Implement `list_worldbank_available_countries()` (coverage check)
- [x] Add country/date filtering to all functions
- [ ] Implement `list_worldbank_indicators()` (discovery) - deferred for future

### Phase 3: Integration ✅
- [x] Add exports to package `__init__.py`
- [x] Update `.gitignore` for cached data
- [x] Basic testing with GDP, HCI, and Trade indicators
- [ ] Create comprehensive example analysis in notebook
  - GDP correlation with fitness
  - HCI triangle analysis (Fitness-GDP-HCI)
  - Time-series growth analysis

### Phase 4: Quality Assurance
- [x] Manual testing with different indicators (GDP, HCI, Trade)
- [x] Test with different country/year filters
- [x] Test DataFrame structure consistency
- [x] Test error handling (duplicate data)
- [ ] Add formal unit tests to test suite
- [ ] Update formal documentation (after validation)

## References

### Data Sources
- World Bank Data: https://data.worldbank.org/
- World Bank API Documentation: https://datahelpdesk.worldbank.org/knowledgebase/topics/125589
- GDP per capita indicator: https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
- Human Capital Index: https://www.worldbank.org/en/publication/human-capital
- HCI Indicator: https://data.worldbank.org/indicator/HD.HCI.OVRL

### Academic References
- Hausmann, R., et al. (2014). *The Atlas of Economic Complexity*. MIT Press. (Uses GDP for validation)
- World Bank (2020). *The Human Capital Index 2020 Update: Human Capital in the Time of COVID-19*. World Bank Group.
- Kraay, A. (2019). "The World Bank Human Capital Index: A Guide." *The World Bank Research Observer*, 34(1), 1-33.

### Related CIPs
- CIP-000B: Atlas data access (established the pattern for data loaders)
- Future: CIP for analyzing Fitness-GDP-HCI relationships
