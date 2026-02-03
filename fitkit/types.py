"""Type definitions for fitkit package.

This module defines the core data structures used throughout fitkit,
including the DataBundle (in-memory representation) and DataLoader protocol
(adapter interface for data sources).
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp

@dataclass
class DataBundle:
    """In-memory representation for bipartite incidence analysis.

    This is the standard interface between data loading and algorithms.
    All algorithms accept DataBundle objects and never perform I/O directly.

    Attributes:
        matrix: Sparse incidence or count matrix (rows × columns)
        row_labels: Labels for matrix rows (e.g., user IDs, country codes)
        col_labels: Labels for matrix columns (e.g., words, product codes)
        metadata: Provenance and configuration information:
            - source: data source identifier (e.g., "bigquery:wikipedia", "synthetic")
            - cache_path: path to cached data if applicable
            - query_params: parameters used to generate/query the data
            - created_at: timestamp when data was loaded/generated
            - Any other relevant context for reproducibility
    """
    matrix: sp.spmatrix
    row_labels: np.ndarray
    col_labels: np.ndarray
    metadata: dict[str, Any]


class DataLoader(Protocol):
    """Abstract interface for data loading adapters.

    Implementations of this protocol handle I/O, authentication, caching,
    and return a standardized in-memory DataBundle.

    This interface is designed to be compatible with future integration
    into data-oriented architectures (e.g., Lynguine's access layer).
    """

    def load(self) -> DataBundle:
        """Load data and return an in-memory DataBundle.

        Returns:
            DataBundle with matrix, labels, and metadata

        Raises:
            IOError: If data cannot be loaded
            AuthenticationError: If credentials are missing/invalid
        """
        ...
