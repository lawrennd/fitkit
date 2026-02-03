"""Core algorithms for fitness-complexity and economic complexity analysis.

This module provides both scikit-learn-style estimators (classes) and
functional APIs for convenience and backward compatibility.

**Estimators (scikit-learn-style, recommended)**:
- FitnessComplexity: Nonlinear Fitness-Complexity fixed point
- ECI: Spectral ECI/PCI baseline
- SinkhornScaler: Masked Sinkhorn-Knopp / IPF scaling

**Functional APIs (convenience, backward compatibility)**:
- fitness_complexity: Nonlinear Fitness-Complexity fixed point
- compute_eci_pci: Spectral ECI/PCI baseline
- sinkhorn_masked: Masked Sinkhorn-Knopp / IPF scaling

All algorithms accept in-memory sparse matrices and perform no I/O.
"""

# Estimator classes (sklearn-style)
from fitkit.algorithms.fitness import FitnessComplexity
from fitkit.algorithms.eci import ECI
from fitkit.algorithms.sinkhorn import SinkhornScaler

# Functional APIs (backward compatibility)
from fitkit.algorithms.fitness import fitness_complexity
from fitkit.algorithms.eci import compute_eci_pci
from fitkit.algorithms.sinkhorn import sinkhorn_masked

__all__ = [
    # Estimators (sklearn-style, recommended)
    "FitnessComplexity",
    "ECI",
    "SinkhornScaler",
    # Functional APIs (backward compatibility)
    "fitness_complexity",
    "compute_eci_pci",
    "sinkhorn_masked",
]
