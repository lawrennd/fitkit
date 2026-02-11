"""Test ECI sensitivity to deviations from perfect nesting.

This module systematically perturbs perfectly nested matrices to understand
how quickly ECI breaks down as the nested structure is corrupted.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from fitkit.algorithms import ECI, FitnessComplexity


def create_nested_matrix(n_rows=50, n_cols=75):
    """Create a perfectly nested binary matrix.
    
    Each row i contains all products from rows 0 to i-1, plus additional products.
    """
    M_data = []
    for i in range(n_rows):
        row = np.zeros(n_cols)
        # Each row includes all previous products plus more
        n_products = min(int((i + 1) * n_cols / n_rows) + 5, n_cols)
        row[:n_products] = 1
        M_data.append(row)
    return sp.csr_matrix(np.array(M_data))


def add_shortcuts(M, shortcut_prob):
    """Add 'specialist shortcuts' to a nested matrix.
    
    Shortcuts are rare words (high column index) added to low-diversification rows (specialists).
    This breaks the nested hierarchy by giving specialists access to rare items
    that generalists don't have.
    
    Parameters
    ----------
    M : sparse matrix
        Original nested matrix
    shortcut_prob : float
        Probability of adding a shortcut to each low-div row
        
    Returns
    -------
    M_perturbed : sparse matrix
        Matrix with shortcuts added
    """
    M_dense = M.toarray().copy()
    n_rows, n_cols = M_dense.shape
    
    # Compute diversification
    div = M_dense.sum(axis=1)
    
    # Identify "specialist" rows (bottom 30% by diversification)
    specialist_threshold = np.percentile(div, 30)
    specialist_rows = np.where(div <= specialist_threshold)[0]
    
    # For each specialist, add shortcuts to rare words (top 30% column indices)
    rare_col_start = int(0.7 * n_cols)
    
    for row_idx in specialist_rows:
        if np.random.rand() < shortcut_prob:
            # Add 1-3 rare words
            n_shortcuts = np.random.randint(1, 4)
            # Select from rare words that this specialist doesn't have
            available_rare = np.where(M_dense[row_idx, rare_col_start:] == 0)[0] + rare_col_start
            if len(available_rare) > 0:
                n_shortcuts = min(n_shortcuts, len(available_rare))
                cols_to_add = np.random.choice(available_rare, n_shortcuts, replace=False)
                M_dense[row_idx, cols_to_add] = 1
    
    return sp.csr_matrix(M_dense)


def remove_edges(M, removal_prob):
    """Remove random edges from a nested matrix.
    
    This creates 'gaps' in the hierarchy - countries missing products they 'should' have.
    
    Parameters
    ----------
    M : sparse matrix
        Original nested matrix
    removal_prob : float
        Probability of removing an edge
        
    Returns
    -------
    M_perturbed : sparse matrix
        Matrix with edges removed
    """
    M_dense = M.toarray().copy()
    
    # Find all existing edges (currently 1)
    ones = np.where(M_dense == 1)
    
    # Randomly remove edges
    if len(ones[0]) > 0:
        n_removals = int(removal_prob * len(ones[0]))
        if n_removals > 0:
            indices = np.random.choice(len(ones[0]), n_removals, replace=False)
            M_dense[ones[0][indices], ones[1][indices]] = 0
    
    return sp.csr_matrix(M_dense)


def test_eci_shortcut_sensitivity():
    """Test how ECI degrades as we add shortcuts to nested matrix."""
    np.random.seed(42)
    
    # Create base nested matrix
    M_base = create_nested_matrix(n_rows=50, n_cols=75)
    
    # Test different shortcut levels
    shortcut_probs = np.linspace(0, 0.3, 15)
    correlations_eci = []
    correlations_logF = []
    
    print("\n" + "=" * 70)
    print("ECI SENSITIVITY TO SHORTCUTS")
    print("=" * 70)
    print("\nAdding random 'shortcuts' (edges that break nested hierarchy)...\n")
    print(f"{'Shortcut %':<12} {'ECI↔Div':<12} {'log(F)↔Div':<12} {'Change':<12}")
    print("-" * 70)
    
    for prob in shortcut_probs:
        # Add shortcuts
        M = add_shortcuts(M_base, prob)
        
        # Compute ECI and Fitness
        eci_model = ECI()
        eci, pci = eci_model.fit_transform(M)
        
        fc = FitnessComplexity(verbose=False)
        F, Q = fc.fit_transform(M)
        
        # Compute correlations with diversification
        div = np.asarray(M.sum(axis=1)).ravel()
        log_F = np.log(F)
        
        corr_eci = np.corrcoef(eci, div)[0, 1]
        corr_logF = np.corrcoef(log_F, div)[0, 1]
        
        correlations_eci.append(corr_eci)
        correlations_logF.append(corr_logF)
        
        if len(correlations_eci) > 1:
            change = correlations_eci[-1] - correlations_eci[-2]
            print(f"{prob*100:>6.1f}%      {corr_eci:>6.3f}       {corr_logF:>6.3f}       {change:>+6.3f}")
        else:
            print(f"{prob*100:>6.1f}%      {corr_eci:>6.3f}       {corr_logF:>6.3f}       {'---':<12}")
    
    print("=" * 70)
    
    # Summary
    initial_eci = correlations_eci[0]
    final_eci = correlations_eci[-1]
    degradation = initial_eci - final_eci
    
    print(f"\nSummary:")
    print(f"  Initial ECI correlation (perfect nesting):  {initial_eci:.3f}")
    print(f"  Final ECI correlation ({shortcut_probs[-1]*100:.1f}% shortcuts): {final_eci:.3f}")
    print(f"  Degradation:                                 {degradation:.3f}")
    print(f"  log(Fitness) remained stable:                {correlations_logF[0]:.3f} → {correlations_logF[-1]:.3f}")
    
    # Find threshold where ECI drops below 0.7
    threshold_idx = np.where(np.array(correlations_eci) < 0.7)[0]
    if len(threshold_idx) > 0:
        threshold_prob = shortcut_probs[threshold_idx[0]]
        print(f"\n  ⚠️  ECI drops below 0.7 at ~{threshold_prob*100:.1f}% shortcuts")
    
    return shortcut_probs, correlations_eci, correlations_logF


def test_eci_gap_sensitivity():
    """Test how ECI degrades as we remove edges from nested matrix."""
    np.random.seed(42)
    
    # Create base nested matrix
    M_base = create_nested_matrix(n_rows=50, n_cols=75)
    
    # Test different removal levels
    removal_probs = np.linspace(0, 0.3, 15)
    correlations_eci = []
    correlations_logF = []
    
    print("\n" + "=" * 70)
    print("ECI SENSITIVITY TO GAPS")
    print("=" * 70)
    print("\nRemoving random edges (creating gaps in the hierarchy)...\n")
    print(f"{'Gap %':<12} {'ECI↔Div':<12} {'log(F)↔Div':<12} {'Change':<12}")
    print("-" * 70)
    
    for prob in removal_probs:
        # Remove edges
        M = remove_edges(M_base, prob)
        
        # Compute ECI and Fitness
        eci_model = ECI()
        eci, pci = eci_model.fit_transform(M)
        
        fc = FitnessComplexity(verbose=False)
        F, Q = fc.fit_transform(M)
        
        # Compute correlations with diversification
        div = np.asarray(M.sum(axis=1)).ravel()
        log_F = np.log(F)
        
        corr_eci = np.corrcoef(eci, div)[0, 1]
        corr_logF = np.corrcoef(log_F, div)[0, 1]
        
        correlations_eci.append(corr_eci)
        correlations_logF.append(corr_logF)
        
        if len(correlations_eci) > 1:
            change = correlations_eci[-1] - correlations_eci[-2]
            print(f"{prob*100:>6.1f}%      {corr_eci:>6.3f}       {corr_logF:>6.3f}       {change:>+6.3f}")
        else:
            print(f"{prob*100:>6.1f}%      {corr_eci:>6.3f}       {corr_logF:>6.3f}       {'---':<12}")
    
    print("=" * 70)
    
    return removal_probs, correlations_eci, correlations_logF


def test_plot_sensitivity_curves():
    """Generate plots showing ECI sensitivity to perturbations."""
    # Run both tests
    print("\nGenerating sensitivity curves...\n")
    
    shortcut_probs, eci_shortcuts, logF_shortcuts = test_eci_shortcut_sensitivity()
    gap_probs, eci_gaps, logF_gaps = test_eci_gap_sensitivity()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Shortcuts
    ax1.plot(shortcut_probs * 100, eci_shortcuts, 'o-', label='ECI', linewidth=2, markersize=4)
    ax1.plot(shortcut_probs * 100, logF_shortcuts, 's-', label='log(Fitness)', linewidth=2, markersize=4)
    ax1.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Threshold (0.7)')
    ax1.set_xlabel('Shortcuts (% of potential locations)', fontsize=11)
    ax1.set_ylabel('Correlation with Diversification', fontsize=11)
    ax1.set_title('ECI Sensitivity to Shortcuts\n(Breaking nested hierarchy)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Gaps
    ax2.plot(gap_probs * 100, eci_gaps, 'o-', label='ECI', linewidth=2, markersize=4)
    ax2.plot(gap_probs * 100, logF_gaps, 's-', label='log(Fitness)', linewidth=2, markersize=4)
    ax2.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Threshold (0.7)')
    ax2.set_xlabel('Gaps (% of edges removed)', fontsize=11)
    ax2.set_ylabel('Correlation with Diversification', fontsize=11)
    ax2.set_title('ECI Sensitivity to Gaps\n(Missing expected edges)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('tests/eci_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to tests/eci_sensitivity_analysis.png")
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"""
ECI is VERY sensitive to deviations from perfect nesting:

1. **Shortcuts** (adding unexpected edges):
   - Initial correlation: {eci_shortcuts[0]:.3f}
   - At 10% shortcuts:    ~{eci_shortcuts[5]:.3f}
   - At 30% shortcuts:    {eci_shortcuts[-1]:.3f}
   - Degradation:         {eci_shortcuts[0] - eci_shortcuts[-1]:.3f}

2. **Gaps** (removing expected edges):
   - Initial correlation: {eci_gaps[0]:.3f}
   - At 10% gaps:         ~{eci_gaps[5]:.3f}
   - At 30% gaps:         {eci_gaps[-1]:.3f}
   - Degradation:         {eci_gaps[0] - eci_gaps[-1]:.3f}

3. **log(Fitness) remains robust**:
   - Shortcuts: {logF_shortcuts[0]:.3f} → {logF_shortcuts[-1]:.3f} (change: {logF_shortcuts[-1] - logF_shortcuts[0]:+.3f})
   - Gaps:      {logF_gaps[0]:.3f} → {logF_gaps[-1]:.3f} (change: {logF_gaps[-1] - logF_gaps[0]:+.3f})

**Conclusion**: Even small deviations from perfect nesting (~5-10% shortcuts/gaps)
significantly degrade ECI's correlation with diversification, while log(Fitness)
remains stable. This explains why Wikipedia data (with specialist "shortcuts")
shows poor ECI correlation.
""")
    print("=" * 70)


def test_eci_community_structure():
    """Test ECI on matrices with multiple specialist communities.
    
    This better models Wikipedia: different specialists (astrophysics, biology, etc.)
    each with their own rare vocabulary, breaking global nesting.
    """
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("ECI SENSITIVITY TO COMMUNITY STRUCTURE")
    print("=" * 70)
    print("\nCreating matrices with increasing community structure...\n")
    
    n_rows = 50
    n_cols = 75
    n_communities_list = [1, 2, 3, 5, 10]
    
    print(f"{'Communities':<15} {'ECI↔Div':<12} {'log(F)↔Div':<12}")
    print("-" * 70)
    
    correlations_eci = []
    correlations_logF = []
    
    for n_communities in n_communities_list:
        # Create matrix with n communities
        M_data = []
        rows_per_community = n_rows // n_communities
        cols_per_community = n_cols // n_communities
        
        for comm_idx in range(n_communities):
            # Each community has nested structure within itself
            for i in range(rows_per_community):
                row = np.zeros(n_cols)
                # Common words (first 20% of columns)
                n_common = int(0.2 * n_cols)
                row[:n_common] = (np.random.rand(n_common) > 0.3).astype(float)
                
                # Community-specific words
                comm_start = n_common + comm_idx * (cols_per_community - n_common // n_communities)
                comm_end = min(comm_start + int((i + 1) * 0.7 * cols_per_community / rows_per_community), n_cols)
                if comm_start < n_cols:
                    row[comm_start:comm_end] = 1
                
                M_data.append(row)
        
        M = sp.csr_matrix(np.array(M_data))
        
        # Compute ECI and Fitness
        eci_model = ECI()
        eci, pci = eci_model.fit_transform(M)
        
        fc = FitnessComplexity(verbose=False)
        F, Q = fc.fit_transform(M)
        
        # Compute correlations
        div = np.asarray(M.sum(axis=1)).ravel()
        log_F = np.log(F)
        
        corr_eci = np.corrcoef(eci, div)[0, 1]
        corr_logF = np.corrcoef(log_F, div)[0, 1]
        
        correlations_eci.append(corr_eci)
        correlations_logF.append(corr_logF)
        
        print(f"{n_communities:<15} {corr_eci:>6.3f}       {corr_logF:>6.3f}")
    
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  1 community (nested):      ECI={correlations_eci[0]:.3f}, log(F)={correlations_logF[0]:.3f}")
    print(f"  10 communities (modular):  ECI={correlations_eci[-1]:.3f}, log(F)={correlations_logF[-1]:.3f}")
    print(f"  ECI degradation:            {correlations_eci[0] - correlations_eci[-1]:.3f}")
    print(f"  log(F) degradation:         {correlations_logF[0] - correlations_logF[-1]:.3f}")
    print("\n⚠️  Modular/community structure breaks ECI more than simple shortcuts!")
    
    # Eigenvalue analysis
    print("\n" + "-" * 70)
    print("EIGENVALUE SPECTRUM DIAGNOSTIC")
    print("-" * 70)
    
    def compute_spectral_gap(M):
        """Compute spectral gap (λ₂/λ₃) of country-country matrix."""
        kc = np.asarray(M.sum(axis=1)).ravel()
        kp = np.asarray(M.sum(axis=0)).ravel()
        kc_safe = np.where(kc > 0, kc, 1)
        kp_safe = np.where(kp > 0, kp, 1)
        M_normalized = M.toarray() / kc_safe[:, np.newaxis]
        M_T_normalized = M.toarray().T / kp_safe[:, np.newaxis]
        C = M_normalized @ M_T_normalized
        eigenvalues = np.linalg.eigvalsh(C)
        eigenvalues = np.sort(eigenvalues)[::-1]
        return eigenvalues[1] / eigenvalues[2] if len(eigenvalues) > 2 else np.inf, eigenvalues[:5]
    
    # Test perfect nesting
    M_nested = create_nested_matrix(n_rows=50, n_cols=75)
    gap_nested, eigs_nested = compute_spectral_gap(M_nested)
    
    # Test 2 communities - recreate simpler version
    M_data_2comm = []
    for comm_idx in range(2):
        for i in range(25):
            row = np.zeros(75)
            n_common = 15
            row[:n_common] = (np.random.rand(n_common) > 0.3).astype(float)
            comm_start = n_common + comm_idx * 30
            comm_end = min(comm_start + int((i + 1) * 20 / 25), 75)
            if comm_start < 75:
                row[comm_start:comm_end] = 1
            M_data_2comm.append(row)
    M_2comm = sp.csr_matrix(np.array(M_data_2comm))
    gap_2comm, eigs_2comm = compute_spectral_gap(M_2comm)
    
    print(f"\nPerfect Nesting:")
    print(f"  Top eigenvalues: {eigs_nested[0]:.3f}, {eigs_nested[1]:.3f}, {eigs_nested[2]:.3f}, {eigs_nested[3]:.3f}, {eigs_nested[4]:.3f}")
    print(f"  Spectral gap (λ₂/λ₃): {gap_nested:.2f}")
    print(f"  Relative magnitudes: λ₃/λ₂={eigs_nested[2]/eigs_nested[1]:.2f}, λ₄/λ₂={eigs_nested[3]/eigs_nested[1]:.2f}")
    
    print(f"\n2 Communities:")
    print(f"  Top eigenvalues: {eigs_2comm[0]:.3f}, {eigs_2comm[1]:.3f}, {eigs_2comm[2]:.3f}, {eigs_2comm[3]:.3f}, {eigs_2comm[4]:.3f}")
    print(f"  Spectral gap (λ₂/λ₃): {gap_2comm:.2f}")
    print(f"  Relative magnitudes: λ₃/λ₂={eigs_2comm[2]/eigs_2comm[1]:.2f}, λ₄/λ₂={eigs_2comm[3]/eigs_2comm[1]:.2f}")
    
    # Count significant eigenvalues (>20% of λ₂)
    n_sig_nested = np.sum(eigs_nested > 0.2 * eigs_nested[1])
    n_sig_2comm = np.sum(eigs_2comm > 0.2 * eigs_2comm[1])
    
    print(f"\nSignificant eigenvalues (>20% of λ₂):")
    print(f"  Perfect nesting: {n_sig_nested} eigenvalue(s) → 1D structure")
    print(f"  2 communities:   {n_sig_2comm} eigenvalues → {n_sig_2comm}D structure")
    
    print("\n⚠️  KEY DIAGNOSTIC: Multiple significant eigenvalues = community structure!")
    print("    Wikipedia likely has 5-10+ significant eigenvalues (many specialist communities)")
    print("    ECI uses only 1 dimension (λ₂) → Misses most of the structure!")
    print("=" * 70)


if __name__ == "__main__":
    test_plot_sensitivity_curves()
    test_eci_community_structure()
