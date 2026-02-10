"""Demo: World Trade Data (1998-2000) with Fitness-Complexity.

This example demonstrates loading the world trade dataset from the R
economiccomplexity package and computing Fitness-Complexity scores.
"""

import pandas as pd
from fitkit.datasets import load_world_trade_1998_2000
from fitkit.algorithms import fitness_complexity

# Load data
print("Loading world trade data (1998-2000)...")
M, countries, products = load_world_trade_1998_2000()

print(f"\nDataset summary:")
print(f"  Countries: {M.shape[0]}")
print(f"  Products:  {M.shape[1]}")
print(f"  Density:   {M.nnz / (M.shape[0] * M.shape[1]):.4f}")
print(f"  Nonzeros:  {M.nnz}")

# Compute fitness-complexity
print("\nComputing Fitness-Complexity...")
F, Q, history = fitness_complexity(M, n_iter=200, tol=1e-10)

print(f"Converged in {len(history['dF'])} iterations")

# Show results
countries = countries.copy()
products = products.copy()
countries['fitness'] = F
products['complexity'] = Q

print("\n" + "="*70)
print("TOP 20 FITTEST COUNTRIES (Economic Fitness)")
print("="*70)
top_countries = countries.nlargest(20, 'fitness')[['country', 'fitness']]
for i, (idx, row) in enumerate(top_countries.iterrows(), 1):
    print(f"{i:2d}. {row['country']:>3s}  Fitness = {row['fitness']:.6f}")

print("\n" + "="*70)
print("TOP 20 MOST COMPLEX PRODUCTS")
print("="*70)
top_products = products.nlargest(20, 'complexity').reset_index(drop=True)
for i in range(len(top_products)):
    print(f"{i+1:2d}. {top_products.loc[i, 'product']}  Complexity = {top_products.loc[i, 'complexity']:.6f}")

print("\n" + "="*70)
print("BOTTOM 20 LEAST COMPLEX PRODUCTS")
print("="*70)
bottom_products = products.nsmallest(20, 'complexity').reset_index(drop=True)
for i in range(len(bottom_products)):
    print(f"{i+1:2d}. {bottom_products.loc[i, 'product']}  Complexity = {bottom_products.loc[i, 'complexity']:.6f}")
