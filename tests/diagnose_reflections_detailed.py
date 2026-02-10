"""Detailed diagnostic: Why reflections doesn't exactly match eigenvalues."""

import numpy as np
import scipy.sparse as sp

from fitkit.algorithms.eci import compute_eci_pci


def create_nested_matrix(n_countries=20, n_products=30):
    """Create perfectly nested matrix."""
    M = sp.lil_matrix((n_countries, n_products))
    for i in range(n_countries):
        n_prods = min(i + 5, n_products)
        M[i, :n_prods] = 1
    return M.tocsr()


def manual_reflections_trace():
    """Manually trace reflections iterations and compare to eigenvalues."""
    M = create_nested_matrix(n_countries=20, n_products=30)
    Mv = M.toarray()
    
    # Filter
    kc = Mv.sum(axis=1)
    kp = Mv.sum(axis=0)
    keep_c = kc > 0
    keep_p = kp > 0
    Mv = Mv[keep_c][:, keep_p]
    kc = kc[keep_c]
    kp = kp[keep_p]
    
    print("="*70)
    print("MANUAL REFLECTIONS ITERATION TRACE")
    print("="*70)
    print(f"Matrix shape: {Mv.shape}")
    
    # Get eigenvalue solution
    Dc_inv = np.diag(1.0 / kc)
    Dp_inv = np.diag(1.0 / kp)
    C = Dc_inv @ Mv @ Dp_inv @ Mv.T
    
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    
    eci_eig_raw = evecs[:, 1]
    
    # Fix sign
    if np.corrcoef(eci_eig_raw, kc)[0, 1] < 0:
        eci_eig_raw = -eci_eig_raw
    
    eci_eig = (eci_eig_raw - eci_eig_raw.mean()) / eci_eig_raw.std(ddof=0)
    
    print(f"\nEigenvalue solution (second eigenvector):")
    print(f"  Raw: mean={eci_eig_raw.mean():.6f}, std={eci_eig_raw.std(ddof=0):.6f}")
    print(f"  Standardized: mean={eci_eig.mean():.6f}, std={eci_eig.std(ddof=0):.6f}")
    
    # Manual reflections
    print(f"\nReflections iterations:")
    k_c = kc.copy()
    k_p = kp.copy()
    
    for it in range(15):
        k_c_new = (Mv @ k_p) / kc
        k_p_new = (Mv.T @ k_c_new) / kp
        
        # Normalize for comparison
        k_c_normed = (k_c_new - k_c_new.mean()) / (k_c_new.std(ddof=0) + 1e-12)
        
        # Fix sign
        if np.corrcoef(k_c_normed, kc)[0, 1] < 0:
            k_c_normed = -k_c_normed
        
        corr = np.corrcoef(k_c_normed, eci_eig)[0, 1]
        delta = np.linalg.norm(k_c_new - k_c) / (np.linalg.norm(k_c_new) + 1e-12)
        
        print(f"  Iter {it+1:3d}: corr={corr:.6f}, delta={delta:.2e}, "
              f"mean={k_c_new.mean():.4f}, std={k_c_new.std(ddof=0):.4f}")
        
        if delta < 1e-6:
            print(f"    → Converged at iteration {it+1}")
            break
        
        k_c = k_c_new
        k_p = k_p_new
    
    print(f"\nFinal comparison:")
    print(f"  Correlation with eigenvalue solution: {corr:.6f}")
    print(f"  Difference suggests: {'Minor numerical precision' if corr > 0.995 else 'Significant algorithmic difference'}")
    
    # Test different normalizations
    print(f"\n" + "-"*70)
    print("TESTING DIFFERENT NORMALIZATIONS:")
    print("-"*70)
    
    # What if we don't normalize by mean/std during iteration?
    k_c = kc.copy()
    k_p = kp.copy()
    for it in range(50):
        k_c_new = (Mv @ k_p) / kc
        k_p_new = (Mv.T @ k_c_new) / kp
        delta = np.linalg.norm(k_c_new - k_c) / (np.linalg.norm(k_c_new) + 1e-12)
        if delta < 1e-8:
            break
        k_c = k_c_new
        k_p = k_p_new
    
    # Now standardize at end
    k_c_final_normed = (k_c_new - k_c_new.mean()) / k_c_new.std(ddof=0)
    if np.corrcoef(k_c_final_normed, kc)[0, 1] < 0:
        k_c_final_normed = -k_c_final_normed
    
    corr_final = np.corrcoef(k_c_final_normed, eci_eig)[0, 1]
    print(f"After {it+1} iterations (norm at end): corr={corr_final:.6f}")


if __name__ == "__main__":
    manual_reflections_trace()
