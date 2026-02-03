"""Data loading adapters for fitkit.

This module provides:
- WikipediaLoader: Load Wikipedia edit data via BigQuery (with caching)
- SyntheticLoader: Generate synthetic data for offline testing
- QueryConfig: Configuration dataclass for data queries
- create_small_fixture: Small deterministic fixture for unit tests

All loaders conform to the DataLoader protocol and return DataBundle objects.
"""

from fitkit.data.fixtures import SyntheticLoader, create_small_fixture
from fitkit.data.loaders import WikipediaLoader, QueryConfig
__all__ = [
    "WikipediaLoader",
    "QueryConfig",
    "SyntheticLoader",
    "create_small_fixture",
]
