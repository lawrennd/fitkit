"""Core algorithms for fitness-complexity and economic complexity analysis.

This module provides scikit-learn-style estimators (classes) as the primary API,
with deprecated functional APIs maintained for backward compatibility.

**Estimators (scikit-learn-style, recommended)**:
- FitnessComplexity: Nonlinear Fitness-Complexity fixed point
- ECI: Spectral ECI/PCI baseline (direct eigenvalue decomposition)
- ECIReflections: Iterative Method of Reflections (use with caution!)
- SinkhornScaler: Masked Sinkhorn-Knopp / IPF scaling

**Functional APIs (DEPRECATED - use classes instead)**:
- fitness_complexity: Use FitnessComplexity class instead (DEPRECATED)
- compute_eci_pci: Use ECI class instead
- compute_eci_pci_reflections: Use ECIReflections class instead (DEPRECATED)
- check_eigengap: Use ECIReflections.check_eigengap() instead (DEPRECATED)
- sinkhorn_masked: Use SinkhornScaler class instead

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
