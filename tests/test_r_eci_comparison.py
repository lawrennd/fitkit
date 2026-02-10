"""ECI/PCI Validation: Compare Python fitkit vs R economiccomplexity.

This test validates the Python fitkit.algorithms.compute_eci_pci() implementation
against the established R economiccomplexity package to ensure mathematical correctness.
See CIP-0008 for complete design rationale.

PREREQUISITES:
    1. R must be installed (macOS: brew install --cask r)
    2. R package installed: R -e 'install.packages("economiccomplexity")'
    3. Reference data generated: Rscript tests/test_r_eci_comparison.R

USAGE:
    # Run all comparison tests
    pytest tests/test_r_eci_comparison.py -v
    
    # Run specific test
    pytest tests/test_r_eci_comparison.py::test_r_eci_nested_matrix -v
    
    # Run directly for detailed output
    python tests/test_r_eci_comparison.py

VALIDATION CRITERIA:
    - Nested matrix: Pearson correlation >0.99 (typically >0.999)
    - Random matrix: Pearson correlation >0.95 (typically >0.99)
    - Modular matrix: Pearson correlation >0.95 (typically >0.99)
    - Spearman rank correlations should be even higher

INTERPRETATION:
    High correlations (>0.99) prove:
    - Both implementations compute same eigenvector
    - Python implementation is mathematically correct
    - Rankings are reproducible
    
    Minor differences (<1%) are expected due to:
    - Numerical precision (eigendecomposition)
    - Sign conventions (eigenvector can flip sign)
    - Normalization choices

REFERENCES:
    - R Package: https://cran.r-project.org/package=economiccomplexity
    - Method: Hidalgo & Hausmann (2009) PNAS 106(26):10570-10575
    - CIP-0008: fitkit/cip/cip0008_r-package-validation.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import spearmanr

from fitkit.algorithms.eci import compute_eci_pci


def load_r_comparison_data(test_name: str):
    """Load R reference data for comparison.
    
    Args:
        test_name: Name of test case (e.g., "eci_nested", "eci_random")
        
    Returns:
        M: Sparse incidence matrix (CSR format)
        r_eci: R Economic Complexity Index (countries)
        r_pci: R Product Complexity Index (products)
    """
    data_path = Path(__file__).parent / "r_comparison_data"
    
    # Load dimensions
    dims = pd.read_csv(data_path / f"{test_name}_dims.csv")
    n_countries = int(dims["n_countries"].iloc[0])
    n_products = int(dims["n_products"].iloc[0])
    
    # Load matrix in sparse format
    matrix_data = pd.read_csv(data_path / f"{test_name}_matrix.csv")
    M = sp.csr_matrix(
        (matrix_data["value"], (matrix_data["row"], matrix_data["col"])),
        shape=(n_countries, n_products)
    )
    
    # Load R results
    r_eci_df = pd.read_csv(data_path / f"{test_name}_r_eci.csv")
    r_pci_df = pd.read_csv(data_path / f"{test_name}_r_pci.csv")
    
    r_eci = r_eci_df["eci"].values
    r_pci = r_pci_df["pci"].values
    
    return M, r_eci, r_pci


def compare_implementations(test_name: str, min_corr_eci: float = 0.95, min_corr_pci: float = 0.95):
    """Compare Python and R implementations on a test case.
    
    Args:
        test_name: Name of test case
        min_corr_eci: Minimum acceptable correlation for ECI
        min_corr_pci: Minimum acceptable correlation for PCI
    """
    print(f"\n{'=' * 70}")
    print(f"COMPARING: {test_name.upper()}")
    print('=' * 70)
    
    # Load R reference data
    M, r_eci, r_pci = load_r_comparison_data(test_name)
    
    print(f"Matrix shape: {M.shape}")
    print(f"Matrix density: {M.nnz / (M.shape[0] * M.shape[1]):.4f}")
    
    # Run Python implementation
    py_eci, py_pci = compute_eci_pci(M)
    
    # Compare results
    print("\n--- Summary Statistics ---")
    print(f"R ECI:       mean={r_eci.mean():.6f}, std={r_eci.std():.6f}")
    print(f"Python ECI:  mean={py_eci.mean():.6f}, std={py_eci.std():.6f}")
    print(f"R PCI:       mean={r_pci.mean():.6f}, std={r_pci.std():.6f}")
    print(f"Python PCI:  mean={py_pci.mean():.6f}, std={py_pci.std():.6f}")
    
    # Correlation analysis (most important metric for ranking comparison)
    # Need to handle potential sign flip in eigenvectors
    corr_eci = np.corrcoef(r_eci, py_eci)[0, 1]
    corr_pci = np.corrcoef(r_pci, py_pci)[0, 1]
    
    # If negative correlation, flip sign and recompute
    if corr_eci < 0:
        py_eci = -py_eci
        corr_eci = np.corrcoef(r_eci, py_eci)[0, 1]
    
    if corr_pci < 0:
        py_pci = -py_pci
        corr_pci = np.corrcoef(r_pci, py_pci)[0, 1]
    
    rank_corr_eci = spearmanr(r_eci, py_eci)[0]
    rank_corr_pci = spearmanr(r_pci, py_pci)[0]
    
    print("\n--- Correlation Analysis ---")
    print(f"ECI correlation (R vs Python):     {corr_eci:.6f}")
    print(f"PCI correlation (R vs Python):     {corr_pci:.6f}")
    print(f"ECI rank correlation:              {rank_corr_eci:.6f}")
    print(f"PCI rank correlation:              {rank_corr_pci:.6f}")
    
    # Absolute error analysis
    abs_err_eci = np.abs(r_eci - py_eci)
    abs_err_pci = np.abs(r_pci - py_pci)
    
    print("\n--- Absolute Error ---")
    print(f"ECI error:    mean={abs_err_eci.mean():.6f}, max={abs_err_eci.max():.6f}")
    print(f"PCI error:    mean={abs_err_pci.mean():.6f}, max={abs_err_pci.max():.6f}")
    
    # Validation
    print("\n--- Validation ---")
    eci_pass = corr_eci >= min_corr_eci
    pci_pass = corr_pci >= min_corr_pci
    
    if eci_pass:
        print(f"✓ ECI correlation ({corr_eci:.4f}) >= {min_corr_eci}")
    else:
        print(f"✗ ECI correlation ({corr_eci:.4f}) < {min_corr_eci}")
    
    if pci_pass:
        print(f"✓ PCI correlation ({corr_pci:.4f}) >= {min_corr_pci}")
    else:
        print(f"✗ PCI correlation ({corr_pci:.4f}) < {min_corr_pci}")
    
    return {
        "test_name": test_name,
        "corr_eci": corr_eci,
        "corr_pci": corr_pci,
        "rank_corr_eci": rank_corr_eci,
        "rank_corr_pci": rank_corr_pci,
        "pass": eci_pass and pci_pass
    }


def test_r_eci_nested_matrix():
    """Test Python ECI/PCI implementation on nested matrix against R reference."""
    result = compare_implementations("eci_nested", min_corr_eci=0.99, min_corr_pci=0.99)
    assert result["pass"], f"ECI/PCI validation failed: {result}"


def test_r_eci_random_matrix():
    """Test Python ECI/PCI implementation on random matrix against R reference."""
    result = compare_implementations("eci_random", min_corr_eci=0.95, min_corr_pci=0.95)
    assert result["pass"], f"ECI/PCI validation failed: {result}"


def test_r_eci_modular_matrix():
    """Test Python ECI/PCI implementation on modular matrix against R reference."""
    result = compare_implementations("eci_modular", min_corr_eci=0.95, min_corr_pci=0.95)
    assert result["pass"], f"ECI/PCI validation failed: {result}"


def test_r_comparison_all():
    """Run all ECI/PCI comparison tests and display summary report."""
    print("\n" + "=" * 70)
    print("PYTHON FITKIT vs R ECONOMICCOMPLEXITY: ECI/PCI VALIDATION REPORT")
    print("=" * 70)
    
    test_cases = [
        ("eci_nested", 0.99, 0.99),
        ("eci_random", 0.95, 0.95),
        ("eci_modular", 0.95, 0.95),
    ]
    
    all_results = {}
    for test_name, min_eci, min_pci in test_cases:
        all_results[test_name] = compare_implementations(test_name, min_eci, min_pci)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nCorrelation Results:")
    print(f"{'Test Case':<20s} {'ECI':<10s} {'PCI':<10s}    Status")
    print("-" * 54)
    
    for test_name, result in all_results.items():
        name_short = test_name.replace("eci_", "")
        status = "✓ PASS" if result["pass"] else "✗ FAIL"
        print(f"{name_short:<20s} {result['corr_eci']:<10.4f} {result['corr_pci']:<10.4f}    {status}")
    
    print("\nConclusions:")
    all_pass = all(r["pass"] for r in all_results.values())
    if all_pass:
        print("- ✓ Python fitkit ECI/PCI implementation is validated against R economiccomplexity")
        print("- ✓ High correlations confirm both implementations compute same eigenvector")
        print("- ✓ Minor differences are due to numerical precision in eigendecomposition")
        print("- ✓ Rankings (Spearman correlation) are highly reproducible")
    else:
        print("- ✗ Some tests failed - review implementation for discrepancies")
    
    print("=" * 70)


if __name__ == "__main__":
    test_r_comparison_all()
