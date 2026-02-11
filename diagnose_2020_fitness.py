"""Diagnose issues with 2020 fitness scores in atlas data."""

import numpy as np
import pandas as pd
from fitkit import load_atlas_trade
from fitkit.algorithms import FitnessComplexity

print("="*70)
print("DIAGNOSING 2020 FITNESS COMPUTATION")
print("="*70)

# Load 2020 data
print("\nLoading 2020 data...")
M_2020, countries_2020, products_2020 = load_atlas_trade(
    year=2020,
    classification='hs92',
    product_level=4,
    rca_threshold=1.0
)

print(f"\nMatrix shape: {M_2020.shape}")
print(f"Matrix density: {M_2020.nnz / (M_2020.shape[0] * M_2020.shape[1]):.4f}")
print(f"Non-zero entries: {M_2020.nnz:,}")

# Check for problematic structure
print("\n" + "="*70)
print("MATRIX DIAGNOSTICS")
print("="*70)

# Check diversification
diversification = np.asarray(M_2020.sum(axis=1)).ravel()
ubiquity = np.asarray(M_2020.sum(axis=0)).ravel()

print(f"\nDiversification (products per country):")
print(f"  Min: {diversification.min():.0f}")
print(f"  Max: {diversification.max():.0f}")
print(f"  Mean: {diversification.mean():.1f}")
print(f"  Median: {np.median(diversification):.1f}")

print(f"\nUbiquity (countries per product):")
print(f"  Min: {ubiquity.min():.0f}")
print(f"  Max: {ubiquity.max():.0f}")
print(f"  Mean: {ubiquity.mean():.1f}")
print(f"  Median: {np.median(ubiquity):.1f}")

# Check for isolated nodes
n_isolated_countries = (diversification == 0).sum()
n_isolated_products = (ubiquity == 0).sum()
print(f"\nIsolated nodes:")
print(f"  Countries with 0 products: {n_isolated_countries}")
print(f"  Products with 0 countries: {n_isolated_products}")

# Check for very low diversification countries
low_div_threshold = 10
n_low_div = (diversification < low_div_threshold).sum()
print(f"\nCountries with < {low_div_threshold} products: {n_low_div}")
if n_low_div > 0:
    low_div_countries = countries_2020[diversification < low_div_threshold]
    print("  Examples:")
    for _, row in low_div_countries.head(10).iterrows():
        idx = countries_2020[countries_2020['country'] == row['country']].index[0]
        print(f"    {row['country']}: {int(diversification[idx])} products")

# Compute fitness
print("\n" + "="*70)
print("COMPUTING FITNESS")
print("="*70)

fc = FitnessComplexity(n_iter=500, tol=1e-10, verbose=True)
F_2020, Q_2020 = fc.fit_transform(M_2020)

print(f"\nConverged: {fc.converged_}")
print(f"Iterations: {fc.n_iter_}")

# Analyze fitness distribution
print("\n" + "="*70)
print("FITNESS DISTRIBUTION")
print("="*70)

print(f"\nFitness statistics:")
print(f"  Min: {F_2020.min():.6f}")
print(f"  Max: {F_2020.max():.6f}")
print(f"  Mean: {F_2020.mean():.6f}")
print(f"  Median: {np.median(F_2020):.6f}")
print(f"  Std: {F_2020.std():.6f}")

# Find extreme values
countries_2020['fitness'] = F_2020
countries_2020['diversification'] = diversification

print("\n" + "="*70)
print("TOP 20 COUNTRIES BY FITNESS")
print("="*70)
top_20 = countries_2020.nlargest(20, 'fitness')
print(f"\n{'Rank':<6} {'Country':<10} {'Fitness':<15} {'Diversification':<15}")
print("-"*50)
for rank, (_, row) in enumerate(top_20.iterrows(), 1):
    print(f"{rank:<6} {row['country']:<10} {row['fitness']:<15.6f} {int(row['diversification']):<15}")

print("\n" + "="*70)
print("BOTTOM 20 COUNTRIES BY FITNESS")
print("="*70)
bottom_20 = countries_2020.nsmallest(20, 'fitness')
print(f"\n{'Rank':<6} {'Country':<10} {'Fitness':<15} {'Diversification':<15}")
print("-"*50)
for rank, (_, row) in enumerate(bottom_20.iterrows(), 1):
    print(f"{rank:<6} {row['country']:<10} {row['fitness']:<15.6f} {int(row['diversification']):<15}")

# Check if Germany is the issue
print("\n" + "="*70)
print("GERMANY SPECIFICALLY")
print("="*70)
germany = countries_2020[countries_2020['country'] == 'DEU']
if not germany.empty:
    ger_idx = germany.index[0]
    print(f"Germany (DEU):")
    print(f"  Fitness: {F_2020[ger_idx]:.6f}")
    print(f"  Diversification: {int(diversification[ger_idx])}")
    print(f"  Rank: {(F_2020 > F_2020[ger_idx]).sum() + 1} out of {len(F_2020)}")
    
    # Compare to other major economies
    major_economies = ['USA', 'CHN', 'JPN', 'GBR', 'FRA', 'ITA', 'KOR']
    print(f"\nComparison to other major economies:")
    print(f"{'Country':<10} {'Fitness':<15} {'Diversification':<15}")
    print("-"*40)
    for code in major_economies:
        country_data = countries_2020[countries_2020['country'] == code]
        if not country_data.empty:
            idx = country_data.index[0]
            print(f"{code:<10} {F_2020[idx]:<15.6f} {int(diversification[idx]):<15}")
else:
    print("Germany (DEU) not found in 2020 data!")

# Check ratio of max to median fitness
max_fitness = F_2020.max()
median_fitness = np.median(F_2020)
ratio = max_fitness / median_fitness
print(f"\n" + "="*70)
print(f"ANOMALY CHECK")
print(f"="*70)
print(f"\nMax/Median fitness ratio: {ratio:.1f}")
if ratio > 100:
    print(f"⚠️  WARNING: Extremely high max/median ratio suggests numerical issues or outliers!")
    print(f"   (Typical ratio should be < 20 for real-world data)")
