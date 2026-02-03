"""Core algorithms for fitness-complexity and economic complexity analysis.

This module provides:
- fitness_complexity: Nonlinear Fitness-Complexity fixed point
- compute_eci_pci: Spectral ECI/PCI baseline
- sinkhorn_masked: Masked Sinkhorn-Knopp / IPF scaling

All algorithms accept in-memory sparse matrices and perform no I/O.
"""

from fitkit.algorithms.fitness import fitness_complexity
from fitkit.algorithms.eci import compute_eci_pci
from fitkit.algorithms.sinkhorn import sinkhorn_masked

__all__ = [
    "fitness_complexity",
    "compute_eci_pci",
    "sinkhorn_masked",
]
