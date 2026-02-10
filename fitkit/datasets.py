"""Dataset loaders for fitkit examples and demos.

This module provides convenient loaders for standard economic complexity datasets,
including the world trade data from the R economiccomplexity package and the
Harvard Atlas of Economic Complexity.
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, Optional, Literal
from urllib.request import urlopen, Request
from urllib.parse import urlparse

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


def _download_atlas_file(url: str, dest_path: Path, desc: str = "file") -> None:
    """Download a file from URL with progress indication.
    
    Handles Dataverse API redirects to S3 signed URLs.
    """
    if dest_path.exists():
        return
    
    print(f"Downloading {desc}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create request with user agent to avoid 403 errors
    req = Request(url, headers={'User-Agent': 'fitkit/1.0 (Python)'})
    
    # Download with proper redirect handling
    with urlopen(req) as response:
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            downloaded = 0
            chunk_size = 8192
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                # Show progress for large files
                if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Every MB
                    progress = (downloaded / total_size) * 100
                    print(f"  Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({progress:.1f}%)")
    
    print(f"Downloaded to {dest_path}")


def _compute_balassa_index(trade_df: pd.DataFrame, 
                           threshold: float = 1.0) -> Tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Compute binary Balassa index (RCA) from trade data.
    
    Args:
        trade_df: DataFrame with columns ['location_code', 'product_code', 'export_value']
        threshold: RCA threshold for binarization (default: 1.0)
        
    Returns:
        M: Binary Balassa index matrix, sparse CSR format
        countries: DataFrame with columns ['idx', 'country']
        products: DataFrame with columns ['idx', 'product']
    """
    # Compute total exports per country and per product
    country_totals = trade_df.groupby('location_code')['export_value'].sum()
    product_totals = trade_df.groupby('product_code')['export_value'].sum()
    total_trade = trade_df['export_value'].sum()
    
    # Compute RCA = (X_cp / X_c) / (X_p / X_total)
    trade_df = trade_df.copy()
    trade_df['country_share'] = trade_df['location_code'].map(country_totals)
    trade_df['product_share'] = trade_df['product_code'].map(product_totals)
    trade_df['rca'] = (trade_df['export_value'] / trade_df['country_share']) / (trade_df['product_share'] / total_trade)
    
    # Binarize
    trade_df['balassa'] = (trade_df['rca'] >= threshold).astype(int)
    
    # Create index mappings
    countries = pd.DataFrame({
        'country': sorted(trade_df['location_code'].unique())
    }).reset_index(names='idx')
    
    products = pd.DataFrame({
        'product': sorted(trade_df['product_code'].unique())
    }).reset_index(names='idx')
    
    country_to_idx = dict(zip(countries['country'], countries['idx']))
    product_to_idx = dict(zip(products['product'], products['idx']))
    
    # Build sparse matrix from Balassa entries
    balassa_data = trade_df[trade_df['balassa'] == 1].copy()
    balassa_data['country_idx'] = balassa_data['location_code'].map(country_to_idx)
    balassa_data['product_idx'] = balassa_data['product_code'].map(product_to_idx)
    
    M = sp.csr_matrix(
        (balassa_data['balassa'].values, 
         (balassa_data['country_idx'].values, balassa_data['product_idx'].values)),
        shape=(len(countries), len(products))
    )
    
    return M, countries, products


def load_atlas_trade(
    year: int = 2000,
    classification: Literal['hs92', 'sitc'] = 'hs92',
    product_level: int = 4,
    rca_threshold: float = 1.0,
    auto_download: bool = True
) -> Tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Load world trade data from Harvard Atlas of Economic Complexity.
    
    Downloads and caches trade data from the Harvard Growth Lab's Atlas of Economic
    Complexity dataset, computes the Balassa index (RCA), and returns a binary
    country-product incidence matrix.
    
    Args:
        year: Year of trade data (1995-2023 typically available for HS92; 1962-2023 for SITC)
        classification: Product classification system (currently supported: 'hs92', 'sitc')
            - 'hs92': Harmonized System 1992 (1995-2023)
            - 'sitc': Standard International Trade Classification (1962-2023)
        product_level: Digit level for product aggregation (2, 4, or 6 for HS92; 4 for SITC)
        rca_threshold: Threshold for binarizing RCA (default: 1.0)
        auto_download: If True, automatically download data if not cached (~50-200 MB per classification)
        
    Returns:
        M: Binary Balassa index matrix (countries × products), sparse CSR format
        countries: DataFrame with columns ['idx', 'country'] mapping matrix rows to country codes
        products: DataFrame with columns ['idx', 'product'] mapping matrix columns to product codes
        
    Examples:
        >>> from fitkit.datasets import load_atlas_trade
        >>> from fitkit.algorithms import fitness_complexity
        >>> 
        >>> # Load HS92 data for 2010
        >>> M, countries, products = load_atlas_trade(year=2010, classification='hs92')
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
        - Data is cached locally after first download
        - First call will download ~50-200MB depending on classification
        - Coverage varies by year and classification
        - Product codes are zero-padded to specified digit level
        
    References:
        The Growth Lab at Harvard University. The Atlas of Economic Complexity.
        https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/T4CHWJ
        
        Hausmann, R., Hidalgo, C. A., Bustos, S., Coscia, M., Simoes, A., & Yildirim, M. A. (2014).
        The Atlas of Economic Complexity: Mapping Paths to Prosperity. MIT Press.
    """
    data_dir = get_data_dir() / "atlas"
    data_dir.mkdir(exist_ok=True)
    
    # Use Harvard Dataverse API for direct file access
    # File IDs from the Atlas datasets on Dataverse
    base_url = "https://dataverse.harvard.edu/api/access/datafile"
    
    # File IDs for different classifications at 4-digit level (country-product-year files)
    # These IDs are from the latest version of the datasets on Dataverse
    file_ids = {
        'hs92': 13439447,   # hs92_country_product_year_4.csv
        'hs96': None,        # Not yet available on Dataverse
        'hs02': None,        # Not yet available on Dataverse  
        'hs07': None,        # Not yet available on Dataverse
        'sitc': 7266079,     # sitc_country_product_year_4.csv (approximate)
    }
    
    file_id = file_ids.get(classification)
    if file_id is None:
        raise ValueError(
            f"Classification {classification} not yet available for auto-download. "
            f"Currently supported: hs92, sitc"
        )
    
    file_url = f"{base_url}/{file_id}"
    
    # Download and cache the data
    cache_file = data_dir / f"atlas_{classification}_country_product_year.csv"
    
    if not cache_file.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"Data file not found: {cache_file}\n"
                f"Set auto_download=True to download automatically from Harvard Dataverse"
            )
        _download_atlas_file(
            file_url,
            cache_file,
            f"Atlas {classification.upper()} country-product-year data (this may take several minutes for ~50-200 MB)"
        )
    
    # Load the data
    print(f"Loading {classification.upper()} data for {year}...")
    df = pd.read_csv(cache_file)
    
    # Filter to requested year
    if 'year' not in df.columns:
        raise ValueError(f"Dataset does not contain 'year' column. Columns: {df.columns.tolist()}")
    
    df_year = df[df['year'] == year].copy()
    
    if len(df_year) == 0:
        available_years = sorted(df['year'].unique())
        raise ValueError(
            f"No data available for year {year} in {classification.upper()}.\n"
            f"Available years: {available_years}"
        )
    
    # Rename columns to standard names
    if 'country_iso3_code' in df_year.columns:
        df_year = df_year.rename(columns={'country_iso3_code': 'location_code'})
    elif 'location_code' not in df_year.columns:
        raise ValueError(f"Could not find country column. Available columns: {df_year.columns.tolist()}")
    
    # Aggregate to requested product level
    if classification.startswith('hs'):
        # HS codes: check which column name is used
        if 'product_hs92_code' in df_year.columns:
            product_col = 'product_hs92_code'
        elif 'hs_product_code' in df_year.columns:
            product_col = 'hs_product_code'
        else:
            raise ValueError(f"Could not find HS product column. Available columns: {df_year.columns.tolist()}")
        df_year['product_code'] = df_year[product_col].astype(str).str.zfill(6).str[:product_level]
    else:
        # SITC codes
        if 'product_sitc_code' in df_year.columns:
            product_col = 'product_sitc_code'
        elif 'sitc_product_code' in df_year.columns:
            product_col = 'sitc_product_code'
        else:
            raise ValueError(f"Could not find SITC product column. Available columns: {df_year.columns.tolist()}")
        df_year['product_code'] = df_year[product_col].astype(str).str.zfill(4).str[:product_level]
    
    # Aggregate export values at the chosen product level
    df_year = df_year.groupby(['location_code', 'product_code'], as_index=False)['export_value'].sum()
    
    # Remove rows with zero or missing exports
    df_year = df_year[df_year['export_value'] > 0].copy()
    
    print(f"Computing Balassa index for {len(df_year)} country-product pairs...")
    M, countries, products = _compute_balassa_index(df_year, threshold=rca_threshold)
    
    print(f"Matrix shape: {M.shape[0]} countries × {M.shape[1]} products")
    print(f"Density: {M.nnz / (M.shape[0] * M.shape[1]):.4f}")
    
    return M, countries, products


def list_atlas_available_years(
    classification: Literal['hs92', 'sitc'] = 'hs92',
    auto_download: bool = True
) -> list:
    """List available years in the Atlas dataset for a given classification.
    
    Args:
        classification: Product classification system (currently supported: 'hs92', 'sitc')
        auto_download: If True, download data if not already cached (~50-200 MB)
        
    Returns:
        List of available years (sorted)
        
    Examples:
        >>> from fitkit.datasets import list_atlas_available_years
        >>> years = list_atlas_available_years('hs92')
        >>> print(f"HS92 data available for {len(years)} years: {years[0]}-{years[-1]}")
    """
    data_dir = get_data_dir() / "atlas"
    cache_file = data_dir / f"atlas_{classification}_country_product_year.csv"
    
    if not cache_file.exists():
        if not auto_download:
            return []
        # Trigger download - use a common year that should exist
        print(f"Downloading {classification.upper()} data to check available years...")
        try:
            load_atlas_trade(year=2000, classification=classification, auto_download=True)
        except ValueError:
            # If 2000 doesn't exist, the download should still have happened
            pass
    
    # Load and check years
    df = pd.read_csv(cache_file, usecols=['year'])
    return sorted(df['year'].unique().tolist())
