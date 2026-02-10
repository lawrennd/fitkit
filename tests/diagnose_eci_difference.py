"""Diagnostic script to understand ECI/PCI differences between R and Python."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

from fitkit.algorithms.eci import compute_eci_pci


def diagnose_nested():
    """Diagnose the nested matrix case."""
    data_path = Path("tests/r_comparison_data")
    
    # Load dimensions
    dims = pd.read_csv(data_path / "eci_nested_dims.csv")
    n_countries = int(dims["n_countries"].iloc[0])
    n_products = int(dims["n_products"].iloc[0])
    
    # Load matrix
    matrix_data = pd.read_csv(data_path / "eci_nested_matrix.csv")
    M = sp.csr_matrix(
        (matrix_data["value"], (matrix_data["row"], (matrix_data["col"]))),
        shape=(n_countries, n_products)
    )
    
    # Load R results
    r_eci = pd.read_csv(data_path / "eci_nested_r_eci.csv")["eci"].values
    r_pci = pd.read_csv(data_path / "eci_nested_r_pci.csv")["pci"].values
    
    # Compute Python results
    py_eci, py_pci = compute_eci_pci(M)
    
    # Detailed eigenanalysis
    Mv = M.toarray()
    kc = Mv.sum(axis=1)
    kp = Mv.sum(axis=0)
    
    Dc_inv = np.diag(1.0 / kc)
    Dp_inv = np.diag(1.0 / kp)
    
    C = Dc_inv @ Mv @ Dp_inv @ Mv.T
    
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    
    print("=" * 70)
    print("NESTED MATRIX DIAGNOSTIC")
    print("=" * 70)
    print(f"\nMatrix shape: {M.shape}")
    print(f"\nTop 10 eigenvalues:")
    for i in range(min(10, len(evals))):
        print(f"  λ[{i}] = {evals[i]:.6f}")
    
    print(f"\nEigenvector 1 (used for ECI):")
    print(f"  Min: {evecs[:, 1].min():.6f}, Max: {evecs[:, 1].max():.6f}")
    print(f"  Mean: {evecs[:, 1].mean():.6f}, Std: {evecs[:, 1].std():.6f}")
    
    print(f"\nR ECI:")
    print(f"  Min: {r_eci.min():.6f}, Max: {r_eci.max():.6f}")
    print(f"  Mean: {r_eci.mean():.6f}, Std: {r_eci.std():.6f}")
    
    print(f"\nPython ECI (after standardization):")
    print(f"  Min: {py_eci.min():.6f}, Max: {py_eci.max():.6f}")
    print(f"  Mean: {py_eci.mean():.6f}, Std: {py_eci.std():.6f}")
    
    # Check correlation
    corr = np.corrcoef(r_eci, py_eci)[0, 1]
    print(f"\nCorrelation: {corr:.6f}")
    
    # Check if sign flip helps
    corr_flipped = np.corrcoef(r_eci, -py_eci)[0, 1]
    print(f"Correlation (flipped): {corr_flipped:.6f}")
    
    # Check differences point by point
    diff = np.abs(r_eci - py_eci)
    print(f"\nAbsolute differences:")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Max: {diff.max():.6f}")
    print(f"  Median: {np.median(diff):.6f}")
    
    # Check correlation with diversification
    corr_with_kc_r = np.corrcoef(r_eci, kc)[0, 1]
    corr_with_kc_py = np.corrcoef(py_eci, kc)[0, 1]
    print(f"\nCorrelation with diversification (k_c):")
    print(f"  R:      {corr_with_kc_r:.6f}")
    print(f"  Python: {corr_with_kc_py:.6f}")
    
    # Check if standardization is the issue
    r_eci_std = r_eci.std()
    py_eci_std = py_eci.std()
    print(f"\nStd dev:")
    print(f"  R:      {r_eci_std:.6f}")
    print(f"  Python: {py_eci_std:.6f}")
    
    # Print a few sample values
    print(f"\nSample values (first 5):")
    print("  Index   R ECI    Py ECI    Diff")
    for i in range(min(5, len(r_eci))):
        print(f"  {i:3d}    {r_eci[i]:7.4f}  {py_eci[i]:7.4f}  {r_eci[i]-py_eci[i]:7.4f}")


if __name__ == "__main__":
    diagnose_nested()
