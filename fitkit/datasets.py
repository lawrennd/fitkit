"""Dataset loaders for fitkit examples and demos.

This module provides convenient loaders for standard economic complexity datasets,
including the world trade data from the R economiccomplexity package.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


def get_data_dir() -> Path:
    """Get the path to the fitkit data directory."""
    return Path(__file__).parent / "data"


def load_world_trade_1998_2000() -> Tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Load world trade data (1998-2000 average) with Balassa index.
    
    This dataset contains international trade data averaged over 1998-2000,
    with products classified using SITC Rev. 2. The Balassa index (RCA) has
    been computed and binarized (threshold=1) to create a binary country-product
    incidence matrix.
    
    Returns:
        M: Binary Balassa index matrix (226 countries × 785 products), sparse CSR format
        countries: DataFrame with columns ['idx', 'country'] mapping matrix rows to country codes
        products: DataFrame with columns ['idx', 'product'] mapping matrix columns to product codes
        
    Examples:
        >>> from fitkit.datasets import load_world_trade_1998_2000
        >>> from fitkit.algorithms import fitness_complexity
        >>> 
        >>> # Load data
        >>> M, countries, products = load_world_trade_1998_2000()
        >>> print(f"Matrix shape: {M.shape}")
        >>> print(f"Density: {M.nnz / (M.shape[0] * M.shape[1]):.4f}")
        >>> 
        >>> # Compute fitness-complexity
        >>> F, Q, history = fitness_complexity(M)
        >>> 
        >>> # Show top 10 fittest countries
        >>> countries['fitness'] = F
        >>> print(countries.nlargest(10, 'fitness')[['country', 'fitness']])
        
    Notes:
        - Source: R economiccomplexity package v2.0.0
        - Original data: World Bank / UN Comtrade
        - Period: Average of years 1998, 1999, 2000
        - Matrix density: ~0.18 (18% of country-product pairs have RCA > 1)
        
    References:
        Hausmann, R., Hidalgo, C. A., Bustos, S., Coscia, M., Simoes, A., & Yildirim, M. A. (2014).
        The Atlas of Economic Complexity: Mapping Paths to Prosperity. MIT Press.
    """
    data_dir = get_data_dir()
    
    # Load sparse Balassa index
    balassa_file = data_dir / "world_trade_1998_2000_balassa.csv"
    if not balassa_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {balassa_file}\n"
            f"Run: Rscript scripts/extract_r_demo_data.R"
        )
    
    data = pd.read_csv(balassa_file)
    countries = pd.read_csv(data_dir / "world_trade_1998_2000_countries.csv")
    products = pd.read_csv(data_dir / "world_trade_1998_2000_products.csv")
    
    # Create sparse matrix
    M = sp.csr_matrix(
        (data['value'], (data['country_idx'], data['product_idx'])),
        shape=(len(countries), len(products))
    )
    
    return M, countries, products


def load_world_trade_raw_1998_2000() -> pd.DataFrame:
    """Load raw world trade data (1998-2000 average) in long format.
    
    This is the original trade data before computing the Balassa index,
    containing export values for country-product pairs.
    
    Returns:
        DataFrame with columns ['country', 'product', 'value']
        - country: 3-letter country code (e.g., 'usa', 'jpn', 'deu')
        - product: 4-digit SITC Rev. 2 product code (e.g., '0011', '7849')
        - value: Export value in USD
        
    Examples:
        >>> from fitkit.datasets import load_world_trade_raw_1998_2000
        >>> 
        >>> # Load raw trade data
        >>> trade = load_world_trade_raw_1998_2000()
        >>> print(f"Total observations: {len(trade)}")
        >>> 
        >>> # Show top exporters
        >>> top_exporters = trade.groupby('country')['value'].sum().nlargest(10)
        >>> print(top_exporters)
        
    References:
        Hausmann, R., Hidalgo, C. A., Bustos, S., Coscia, M., Simoes, A., & Yildirim, M. A. (2014).
        The Atlas of Economic Complexity: Mapping Paths to Prosperity. MIT Press.
    """
    data_dir = get_data_dir()
    
    trade_file = data_dir / "world_trade_1998_2000.csv"
    if not trade_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {trade_file}\n"
            f"Run: Rscript scripts/extract_r_demo_data.R"
        )
    
    return pd.read_csv(trade_file)
