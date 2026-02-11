"""Fitkit: Scientific software for fitness-complexity analysis.

This package provides algorithms and data loaders for analyzing bipartite
networks using the Fitness-Complexity method and ECI/PCI baselines.

Main components:
- fitkit.algorithms: Core computational methods (fitness, ECI, Sinkhorn)
- fitkit.community: Community detection and within-community analysis
- fitkit.data: Data loading adapters (Wikipedia/BigQuery, synthetic fixtures)
- fitkit.datasets: Standard datasets for demos and validation
- fitkit.types: Shared type definitions (DataBundle, DataLoader protocol)

Quick start:
    >>> from fitkit.data.loaders import WikipediaLoader, QueryConfig
    >>> from fitkit.algorithms.fitness import fitness_complexity
    >>>
    >>> # Load data
    >>> config = QueryConfig(max_authors=100)
    >>> loader = WikipediaLoader(config, cache_path="data/cache.parquet")
    >>> bundle = loader.load()
    >>>
    >>> # Compute fitness-complexity
    >>> F, Q, history = fitness_complexity(bundle.matrix)

References:
    Lawrence, N.D. (2024). "Conditional Likelihood Interpretation of Economic Fitness".
"""

__version__ = "0.1.0"

# Export key types for convenience
from fitkit.types import DataBundle, DataLoader

# Export dataset loaders
from fitkit.datasets import (
    load_world_trade_1998_2000,
    load_world_trade_raw_1998_2000,
    load_atlas_trade,
    list_atlas_available_years,
    load_atlas_product_names,
    load_worldbank_indicator,
    load_gdp_per_capita,
    load_human_capital_index,
    list_worldbank_available_countries,
    load_country_names,
)

__all__ = [
    "DataBundle",
    "DataLoader",
    "load_world_trade_1998_2000",
    "load_world_trade_raw_1998_2000",
    "load_atlas_trade",
    "list_atlas_available_years",
    "load_atlas_product_names",
    "load_worldbank_indicator",
    "load_gdp_per_capita",
    "load_human_capital_index",
    "list_worldbank_available_countries",
    "load_country_names",
    "__version__",
]
