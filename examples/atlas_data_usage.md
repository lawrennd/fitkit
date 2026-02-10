# Using Atlas of Economic Complexity Data

The `fitkit` package now provides easy access to the Harvard Atlas of Economic Complexity datasets through the `load_atlas_trade()` function.

## Quick Start

```python
from fitkit import load_atlas_trade, list_atlas_available_years
from fitkit.algorithms import fitness_complexity

# See what years are available
years = list_atlas_available_years('hs92')
print(f"Data available for {len(years)} years: {years[0]}-{years[-1]}")

# Load HS92 data for 2010 at 4-digit level
M, countries, products = load_atlas_trade(
    year=2010, 
    classification='hs92',
    product_level=4
)

print(f"Matrix shape: {M.shape[0]} countries × {M.shape[1]} products")
print(f"Density: {M.nnz / (M.shape[0] * M.shape[1]):.4f}")

# Compute fitness-complexity
F, Q, history = fitness_complexity(M)

# Show top 10 fittest countries
countries['fitness'] = F
print("\nTop 10 Countries by Economic Fitness:")
print(countries.nlargest(10, 'fitness')[['country', 'fitness']])
```

## Available Classifications

Currently supported:
- **hs92**: Harmonized System 1992 (1988-2024, most years from 1995 onward)
- **sitc**: Standard International Trade Classification (1962-2023)

## Product Aggregation Levels

- **2-digit**: Very broad categories (~20-40 products)
- **4-digit**: Standard level (recommended, ~90-200 products)
- **6-digit**: Most detailed (thousands of products, sparse matrix)

## First Download

The first time you call `load_atlas_trade()`, it will automatically download the dataset from Harvard Dataverse (~475 MB for HS92). This takes a few minutes but the data is cached locally for subsequent use.

To manually trigger download and see available years:

```python
# This downloads if needed and shows available years
years = list_atlas_available_years('hs92', auto_download=True)
```

## Comparing Different Time Periods

```python
from fitkit import load_atlas_trade
from fitkit.algorithms import fitness_complexity
import matplotlib.pyplot as plt

# Load data for two different years
M_2000, countries_2000, _ = load_atlas_trade(year=2000, classification='hs92')
M_2020, countries_2020, _ = load_atlas_trade(year=2020, classification='hs92')

# Compute fitness for both years
F_2000, _, _ = fitness_complexity(M_2000)
F_2020, _, _ = fitness_complexity(M_2020)

# Add to dataframes
countries_2000['fitness_2000'] = F_2000
countries_2020['fitness_2020'] = F_2020

# Merge and compare
comparison = countries_2000.merge(
    countries_2020[['country', 'fitness_2020']], 
    on='country', 
    how='inner'
)
comparison['fitness_change'] = comparison['fitness_2020'] - comparison['fitness_2000']

# Show biggest gainers
print("\nCountries with largest fitness gains (2000-2020):")
print(comparison.nlargest(10, 'fitness_change')[['country', 'fitness_2000', 'fitness_2020', 'fitness_change']])

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(comparison['fitness_2000'], comparison['fitness_2020'], alpha=0.5)
plt.plot([0, comparison[['fitness_2000', 'fitness_2020']].max().max()], 
         [0, comparison[['fitness_2000', 'fitness_2020']].max().max()], 
         'k--', alpha=0.3)
plt.xlabel('Fitness 2000')
plt.ylabel('Fitness 2020')
plt.title('Economic Fitness: 2000 vs 2020')
plt.tight_layout()
plt.show()
```

## Data Source

Data is sourced from the Harvard Growth Lab's Atlas of Economic Complexity:
- Original data: UN Comtrade
- Cleaned and harmonized by the Harvard Growth Lab
- Available at: https://atlas.hks.harvard.edu/data-downloads

## Citation

When using Atlas data in publications:

```
The Growth Lab at Harvard University, 2024, 
"International Trade Data (HS92), 1988-2024", 
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/T4CHWJ, 
Harvard Dataverse
```

For the original Atlas book:

```
Hausmann, R., Hidalgo, C., Bustos, S., Coscia, M., Chung, S., Jimenez, J., 
Simoes, A., Yildirim, M. (2013). The Atlas of Economic Complexity. 
Cambridge, MA: MIT Press.
```
