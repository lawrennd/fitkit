"""Check actual fitness values and products."""

import numpy as np
from fitkit import load_atlas_trade
from fitkit.algorithms.fitness import _fitness_complexity

# Load 2020 data
M_2020, countries_2020, products_2020 = load_atlas_trade(
    year=2020,
    classification='hs92',
    product_level=4,
    rca_threshold=1.0
)

# Run algorithm
F, Q, history = _fitness_complexity(M_2020, n_iter=500, tol=1e-10, return_history=True, verbose=False)

# Find countries with non-tiny fitness
non_tiny = F > 1e-10
print(f"Countries with fitness > 1e-10: {non_tiny.sum()}")
print(f"\nThese countries:")
for idx in np.where(non_tiny)[0]:
    country_code = countries_2020.iloc[idx]['country']
    print(f"  {country_code}: {F[idx]:.10f}")

# Check for unique products
ubiquity = M_2020.sum(axis=0).A1
unique_products = np.where(ubiquity == 1)[0]
print(f"\nTotal products exported by only 1 country: {len(unique_products)}")

# Find which countries export unique products
print("\nCountries with unique products:")
for idx in np.where(non_tiny)[0]:
    country_code = countries_2020.iloc[idx]['country']
    exports = M_2020[idx, :].toarray().ravel()
    unique_exports = np.where((exports > 0) & (ubiquity == 1))[0]
    if len(unique_exports) > 0:
        print(f"  {country_code}: {len(unique_exports)} unique products")
        # Show some product names
        for prod_idx in unique_exports[:3]:
            prod_code = products_2020.iloc[prod_idx]['product']
            prod_name = products_2020.iloc[prod_idx].get('product_name', 'Unknown')
            print(f"    - {prod_code}: {prod_name}")

# Check raw F before final normalization
print("\n" + "="*70)
print("Checking if normalization is the issue...")
print("="*70)

# The algorithm normalizes at the end so that mean(F) = mean(Q) = 1
# Let's see what F looks like before that final normalization
print(f"\nCurrent F stats:")
print(f"  Mean: {F.mean():.6f}")
print(f"  Min: {F.min():.6f}")
print(f"  Max: {F.max():.6f}")
print(f"  Non-zero (>1e-10): {(F > 1e-10).sum()}")

# The problem is likely that almost all fitness has collapsed to near-zero
# Let's check if this is a log(0) issue
print(f"\nChecking for underflow:")
print(f"  F values < 1e-300: {(F < 1e-300).sum()}")
print(f"  F values exactly 0: {(F == 0).sum()}")

# Let's trace what happens with unique products
print("\n" + "="*70)
print("Understanding unique product dynamics:")
print("="*70)
print("\nIn Fitness-Complexity algorithm:")
print("  - Countries with unique products get initial advantage")
print("  - But this can lead to runaway concentration")
print("  - Other countries' fitness → 0 as algorithm iterates")
print("\nPossible solutions:")
print("  1. Add small epsilon to prevent exact zeros")
print("  2. Filter out countries/products with very low connectivity")
print("  3. Use alternative initialization")
print("  4. Add regularization term")
