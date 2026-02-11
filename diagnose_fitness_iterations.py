"""Diagnose fitness iteration behavior on 2020 data."""

import numpy as np
from fitkit import load_atlas_trade
from fitkit.algorithms.fitness import _fitness_complexity

print("="*70)
print("DETAILED ITERATION DIAGNOSTICS")
print("="*70)

# Load 2020 data
print("\nLoading 2020 data...")
M_2020, countries_2020, products_2020 = load_atlas_trade(
    year=2020,
    classification='hs92',
    product_level=4,
    rca_threshold=1.0
)

print(f"Matrix: {M_2020.shape[0]} countries × {M_2020.shape[1]} products")

# Run algorithm with detailed history
print("\nRunning fitness-complexity with detailed tracking...")
F, Q, history = _fitness_complexity(M_2020, n_iter=500, tol=1e-10, return_history=True, verbose=False)

print(f"\nConverged: {history['converged']}")
print(f"Iterations: {history['n_iter']}")

# Analyze convergence behavior
dF = history['dF']
dQ = history['dQ']

print("\n" + "="*70)
print("CONVERGENCE BEHAVIOR")
print("="*70)

print(f"\nFirst 10 iterations:")
print(f"{'Iter':<6} {'dF':<15} {'dQ':<15} {'max(dF,dQ)':<15}")
print("-"*50)
for i in range(min(10, len(dF))):
    print(f"{i+1:<6} {dF[i]:<15.6e} {dQ[i]:<15.6e} {max(dF[i], dQ[i]):<15.6e}")

if len(dF) > 20:
    print(f"\nLast 10 iterations:")
    print(f"{'Iter':<6} {'dF':<15} {'dQ':<15} {'max(dF,dQ)':<15}")
    print("-"*50)
    for i in range(max(0, len(dF)-10), len(dF)):
        print(f"{i+1:<6} {dF[i]:<15.6e} {dQ[i]:<15.6e} {max(dF[i], dQ[i]):<15.6e}")

# Check for numerical issues
print("\n" + "="*70)
print("NUMERICAL HEALTH CHECK")
print("="*70)

print(f"\nFitness (F):")
print(f"  Non-zero values: {(F > 0).sum()} / {len(F)}")
print(f"  Zero values: {(F == 0).sum()}")
print(f"  NaN values: {np.isnan(F).sum()}")
print(f"  Inf values: {np.isinf(F).sum()}")
print(f"  Mean: {F.mean():.6f}")

print(f"\nComplexity (Q):")
print(f"  Non-zero values: {(Q > 0).sum()} / {len(Q)}")
print(f"  Zero values: {(Q == 0).sum()}")
print(f"  NaN values: {np.isnan(Q).sum()}")
print(f"  Inf values: {np.isinf(Q).sum()}")
print(f"  Mean: {Q.mean():.6f}")

# Check if it's a single outlier
n_nonzero_fitness = (F > 1e-10).sum()
print(f"\nCountries with fitness > 1e-10: {n_nonzero_fitness}")
if n_nonzero_fitness < 10:
    print("  This suggests numerical collapse to a degenerate solution!")
    print("  Expected: Most/all countries should have positive fitness")

# Check Germany's exports
ger_idx = countries_2020[countries_2020['country'] == 'DEU'].index[0]
ger_exports = M_2020[ger_idx, :].toarray().ravel()
print(f"\nGermany's exports:")
print(f"  Exports {ger_exports.sum():.0f} products")
print(f"  Product indices: {np.where(ger_exports > 0)[0][:20].tolist()}... (first 20)")

# Check if Germany exports unique products
ger_unique = (ger_exports > 0) & (ubiquity := M_2020.sum(axis=0).A1 == 1)
n_ger_unique = ger_unique.sum()
print(f"  Products ONLY exported by Germany: {n_ger_unique}")
if n_ger_unique > 0:
    print(f"    ⚠️  This could cause numerical issues!")
