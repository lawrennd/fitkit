"""Eigengap diagnostic tool for Method of Reflections.

This script helps diagnose why Method of Reflections might fail or
converge slowly by analyzing the eigenvalue spectrum of the projection matrix.
"""

import numpy as np
import scipy.sparse as sp

from fitkit.algorithms.eci_reflections import check_eigengap


def create_test_matrices():
    """Create various test matrices with different eigengap properties."""
    
    # 1. Nested (good eigengap)
    M_nested = sp.lil_matrix((20, 30))
    for i in range(20):
        M_nested[i, :min(i+5, 30)] = 1
    
    # 2. Modular/block-diagonal (ZERO eigengap - degenerate!)
    M_modular = sp.lil_matrix((10, 20))
    for b in range(2):
        r_start = b * 5
        r_end = (b+1) * 5
        c_start = b * 10
        c_end = (b+1) * 10
        M_modular[r_start:r_end, c_start:c_end] = 1
    
    # 3. Random sparse (moderate eigengap)
    np.random.seed(42)
    M_random = sp.random(50, 75, density=0.15, format='csr')
    M_random.data[:] = 1
    
    # 4. Nearly complete (small eigengap)
    M_dense = sp.lil_matrix((15, 20))
    M_dense[:, :] = 1
    # Remove a few entries to avoid full rank
    for i in range(5):
        M_dense[i, i] = 0
    
    # 5. Star graph (very small eigengap)
    M_star = sp.lil_matrix((10, 10))
    M_star[0, :] = 1  # Hub connects to all
    for i in range(1, 10):
        M_star[i, i] = 1  # Others connect to themselves only
    
    return {
        'nested': M_nested.tocsr(),
        'modular': M_modular.tocsr(),
        'random': M_random,
        'dense': M_dense.tocsr(),
        'star': M_star.tocsr(),
    }


def run_eigengap_diagnostics():
    """Run eigengap diagnostics on all test matrices."""
    matrices = create_test_matrices()
    
    print("="*70)
    print("EIGENGAP DIAGNOSTIC: METHOD OF REFLECTIONS CONVERGENCE PREDICTOR")
    print("="*70)
    print("\nThis tool predicts whether Method of Reflections will:")
    print("  ✓ Converge quickly (good eigengap)")
    print("  ⚠️  Converge slowly (small eigengap)")
    print("  ✗ Fail completely (degenerate eigenspace)")
    
    results = {}
    for name, M in matrices.items():
        print(f"\n{'='*70}")
        print(f"Matrix: {name.upper()}")
        print(f"{'='*70}")
        
        try:
            info = check_eigengap(M, verbose=True)
            results[name] = info
        except Exception as e:
            print(f"Error: {e}")
            results[name] = None
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Matrix':<15s} {'Eigengap':<12s} {'λ₁/λ₀':<10s} {'Est.Iters':<12s} {'Status'}")
    print("-"*70)
    
    for name, info in results.items():
        if info is None:
            print(f"{name:<15s} {'ERROR':<12s}")
            continue
        
        eigengap_str = f"{info['eigengap']:.6f}"
        ratio_str = f"{info['ratio_lambda1_lambda0']:.6f}"
        iters = info['iterations_for_99pct']
        iters_str = f"~{int(iters)}" if iters < 1000 else "∞"
        
        if info['is_degenerate']:
            status = "✗ DEGENERATE"
        elif iters < 20:
            status = "✓ Fast"
        elif iters < 100:
            status = "⚠️  Slow"
        else:
            status = "⚠️  Very slow"
        
        print(f"{name:<15s} {eigengap_str:<12s} {ratio_str:<10s} {iters_str:<12s} {status}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    print("✓ Use Method of Reflections: nested, random (good eigengap)")
    print("⚠️  Use with caution: dense (slow convergence)")
    print("✗ Don't use: modular, star (degenerate/undefined)")
    print("\nFor problematic cases, use compute_eci_pci() (direct eigenvalue method)")
    print("="*70)


if __name__ == "__main__":
    run_eigengap_diagnostics()
