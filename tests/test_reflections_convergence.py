"""Test Method of Reflections convergence to eigenvalues solution.

This test verifies that the Python Method of Reflections implementation
correctly converges to the same eigenvector as the direct eigenvalue method.
"""

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

from fitkit.algorithms import ECI, ECIReflections


def create_nested_matrix(n_countries=20, n_products=30):
    """Create perfectly nested matrix."""
    M = sp.lil_matrix((n_countries, n_products))
    for i in range(n_countries):
        n_prods = min(i + 5, n_products)
        M[i, :n_prods] = 1
    return M.tocsr()


def create_random_matrix(n_countries=50, n_products=75, density=0.15, seed=42):
    """Create random sparse matrix."""
    np.random.seed(seed)
    M = sp.random(n_countries, n_products, density=density, format='csr')
    M.data[:] = 1
    return M


def test_reflections_convergence_nested():
    """Test that reflections converges to eigenvalues on nested matrix."""
    print("\n" + "="*70)
    print("TEST: Reflections convergence on NESTED matrix")
    print("="*70)
    
    M = create_nested_matrix()
    
    # Check eigengap first
    print("\nStep 1: Eigengap diagnostic")
    gap_info = ECIReflections.check_eigengap(M, verbose=True)
    
    # Compute with both methods
    print("\nStep 2: Computing ECI/PCI")
    eci_model = ECI()
    eci_eig, pci_eig = eci_model.fit_transform(M)
    
    refl_model = ECIReflections(max_iter=50)
    eci_refl, pci_refl = refl_model.fit_transform(M)
    
    print(f"Eigenvalues method: completed")
    print(f"Reflections method: converged={refl_model.converged_}, iterations={refl_model.n_iter_}")
    
    # Compare results
    corr_eci = np.corrcoef(eci_eig, eci_refl)[0, 1]
    corr_pci = np.corrcoef(pci_eig, pci_refl)[0, 1]
    rank_corr_eci = spearmanr(eci_eig, eci_refl)[0]
    rank_corr_pci = spearmanr(pci_eig, pci_refl)[0]
    
    print("\nStep 3: Comparison")
    print(f"ECI correlation (Pearson):  {np.abs(corr_eci):.6f}")
    print(f"PCI correlation (Pearson):  {np.abs(corr_pci):.6f}")
    print(f"ECI correlation (Spearman): {np.abs(rank_corr_eci):.6f}")
    print(f"PCI correlation (Spearman): {np.abs(rank_corr_pci):.6f}")
    
    # Validation
    assert np.abs(corr_eci) > 0.999, f"Reflections did not converge to eigenvalues! corr={corr_eci:.4f}"
    assert np.abs(corr_pci) > 0.999, f"Reflections did not converge to eigenvalues! corr={corr_pci:.4f}"
    
    print("\n✓ PASS: Reflections converged to eigenvalues solution!")
    return True


def test_reflections_convergence_random():
    """Test that reflections converges to eigenvalues on random matrix."""
    print("\n" + "="*70)
    print("TEST: Reflections convergence on RANDOM matrix")
    print("="*70)
    
    M = create_random_matrix()
    
    # Check eigengap
    print("\nEigengap diagnostic:")
    gap_info = ECIReflections.check_eigengap(M, verbose=True)
    
    # Compute
    eci_model = ECI()
    eci_eig, pci_eig = eci_model.fit_transform(M)
    
    refl_model = ECIReflections(max_iter=50)
    eci_refl, pci_refl = refl_model.fit_transform(M)
    
    print(f"\nReflections: converged={refl_model.converged_}, iterations={refl_model.n_iter_}")
    
    # Compare
    corr_eci = np.corrcoef(eci_eig, eci_refl)[0, 1]
    corr_pci = np.corrcoef(pci_eig, pci_refl)[0, 1]
    
    print(f"ECI correlation: {np.abs(corr_eci):.6f}")
    print(f"PCI correlation: {np.abs(corr_pci):.6f}")
    
    assert np.abs(corr_eci) > 0.999
    assert np.abs(corr_pci) > 0.999
    
    print("\n✓ PASS: Reflections converged to eigenvalues solution!")
    return True


def test_reflections_vs_r_eigenvalues():
    """Compare Python reflections to R eigenvalues (should match if Python is correct)."""
    print("\n" + "="*70)
    print("TEST: Python Reflections vs R Eigenvalues")
    print("="*70)
    
    # Load R eigenvalues data
    import pandas as pd
    from pathlib import Path
    
    data_path = Path("tests/r_comparison_data")
    dims = pd.read_csv(data_path / "eci_nested_dims.csv")
    n_countries = int(dims["n_countries"].iloc[0])
    n_products = int(dims["n_products"].iloc[0])
    
    matrix_data = pd.read_csv(data_path / "eci_nested_matrix.csv")
    M = sp.csr_matrix(
        (matrix_data["value"], (matrix_data["row"], matrix_data["col"])),
        shape=(n_countries, n_products)
    )
    
    r_eci = pd.read_csv(data_path / "eci_nested_r_eci.csv")["eci"].values
    r_pci = pd.read_csv(data_path / "eci_nested_r_pci.csv")["pci"].values
    
    # Compute with Python reflections
    refl_model = ECIReflections(max_iter=50, check_eigengap_first=False)
    py_eci, py_pci = refl_model.fit_transform(M)
    
    # Compare
    corr_eci = np.corrcoef(r_eci, py_eci)[0, 1]
    corr_pci = np.corrcoef(r_pci, py_pci)[0, 1]
    
    print(f"\nPython Reflections vs R Eigenvalues:")
    print(f"  ECI correlation: {np.abs(corr_eci):.6f}")
    print(f"  PCI correlation: {np.abs(corr_pci):.6f}")
    
    # Should match R eigenvalues well if theory is correct
    if np.abs(corr_eci) > 0.99:
        print("\n✓ Excellent match - Python reflections works correctly!")
    elif np.abs(corr_eci) > 0.95:
        print("\n✓ Good match - minor numerical differences")
    else:
        print(f"\n⚠️  Poor match (r={corr_eci:.3f}) - implementation issue?")
    
    return np.abs(corr_eci) > 0.95


if __name__ == "__main__":
    print("\n" + "="*70)
    print("METHOD OF REFLECTIONS: CONVERGENCE VALIDATION")
    print("="*70)
    
    try:
        test_reflections_convergence_nested()
    except AssertionError as e:
        print(f"\n✗ FAIL: {e}")
    
    try:
        test_reflections_convergence_random()
    except AssertionError as e:
        print(f"\n✗ FAIL: {e}")
    
    test_reflections_vs_r_eigenvalues()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
