"""Compare 2000 vs 2020 data structure."""

import numpy as np
from fitkit import load_atlas_trade
from fitkit.algorithms.fitness import _fitness_complexity

print("="*70)
print("COMPARING 2000 vs 2020 DATA")
print("="*70)

# Load 2000 data
print("\n--- 2000 Data ---")
M_2000, countries_2000, products_2000 = load_atlas_trade(
    year=2000,
    classification='hs92',
    product_level=4,
    rca_threshold=1.0
)

ubiquity_2000 = M_2000.sum(axis=0).A1
unique_2000 = (ubiquity_2000 == 1).sum()

print(f"Matrix: {M_2000.shape[0]} countries × {M_2000.shape[1]} products")
print(f"Density: {M_2000.nnz / (M_2000.shape[0] * M_2000.shape[1]):.4f}")
print(f"Products with ubiquity=1: {unique_2000}")

# Compute fitness for 2000
F_2000, Q_2000, _ = _fitness_complexity(M_2000, n_iter=500, tol=1e-10, return_history=True, verbose=False)
n_nonzero_2000 = (F_2000 > 1e-10).sum()
print(f"Countries with fitness > 1e-10: {n_nonzero_2000}")
print(f"Max fitness: {F_2000.max():.6f}")
print(f"Median fitness: {np.median(F_2000):.6f}")

# Load 2020 data
print("\n--- 2020 Data ---")
M_2020, countries_2020, products_2020 = load_atlas_trade(
    year=2020,
    classification='hs92',
    product_level=4,
    rca_threshold=1.0
)

ubiquity_2020 = M_2020.sum(axis=0).A1
unique_2020 = (ubiquity_2020 == 1).sum()

print(f"Matrix: {M_2020.shape[0]} countries × {M_2020.shape[1]} products")
print(f"Density: {M_2020.nnz / (M_2020.shape[0] * M_2020.shape[1]):.4f}")
print(f"Products with ubiquity=1: {unique_2020}")

# Compute fitness for 2020
F_2020, Q_2020, _ = _fitness_complexity(M_2020, n_iter=500, tol=1e-10, return_history=True, verbose=False)
n_nonzero_2020 = (F_2020 > 1e-10).sum()
print(f"Countries with fitness > 1e-10: {n_nonzero_2020}")
print(f"Max fitness: {F_2020.max():.6f}")
print(f"Median fitness: {np.median(F_2020):.6f}")

# Summary
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print(f"\n2000 has {unique_2000} products with ubiquity=1")
print(f"2020 has {unique_2020} products with ubiquity=1")

if unique_2000 < unique_2020:
    print("\n⚠️  2020 has more unique products than 2000!")
    print("   This explains why 2020 shows worse numerical collapse.")

if n_nonzero_2020 < n_nonzero_2000:
    print(f"\n⚠️  2020 has numerical collapse: only {n_nonzero_2020} countries with non-tiny fitness")
    print(f"   vs {n_nonzero_2000} countries in 2000")

print("\nRECOMMENDATION:")
print("  Filter out products with ubiquity ≤ N (e.g., N=2 or N=5)")
print("  before running Fitness-Complexity algorithm.")
print("  This is standard practice in the literature.")
