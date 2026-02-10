"""Core algorithms for fitness-complexity and economic complexity analysis.

This module provides both scikit-learn-style estimators (classes) and
functional APIs for convenience and backward compatibility.

**Estimators (scikit-learn-style, recommended)**:
- FitnessComplexity: Nonlinear Fitness-Complexity fixed point
- ECI: Spectral ECI/PCI baseline (direct eigenvalue decomposition)
- ECIReflections: Iterative Method of Reflections (use with caution!)
- SinkhornScaler: Masked Sinkhorn-Knopp / IPF scaling

**Functional APIs (convenience, backward compatibility)**:
- fitness_complexity: Nonlinear Fitness-Complexity fixed point
- compute_eci_pci: Spectral ECI/PCI baseline (direct eigenvalue)
- compute_eci_pci_reflections: Iterative Method of Reflections
- check_eigengap: Diagnostic for reflections convergence
- sinkhorn_masked: Masked Sinkhorn-Knopp / IPF scaling

All algorithms accept in-memory sparse matrices and perform no I/O.
"""

# Estimator classes (sklearn-style)
from fitkit.algorithms.eci import ECI, compute_eci_pci
from fitkit.algorithms.eci_reflections import (
    ECIReflections,
    compute_eci_pci_reflections,
    check_eigengap,
)
from fitkit.algorithms.fitness import FitnessComplexity, fitness_complexity
from fitkit.algorithms.sinkhorn import SinkhornScaler, sinkhorn_masked

__all__ = [
    # Estimators (sklearn-style, recommended)
    "FitnessComplexity",
    "ECI",
    "ECIReflections",
    "SinkhornScaler",
    # Functional APIs (backward compatibility)
    "fitness_complexity",
    "compute_eci_pci",
    "compute_eci_pci_reflections",
    "check_eigengap",
    "sinkhorn_masked",
]
