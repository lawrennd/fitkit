"""Dataset loaders for fitkit examples and demos.

This module provides convenient loaders for standard economic complexity datasets,
including the world trade data from the R economiccomplexity package and the
Harvard Atlas of Economic Complexity, as well as World Bank economic indicators.
"""

import os
import json
import warnings
from pathlib import Path
from typing import Tuple, Optional, Literal, List, Dict, Any
from urllib.request import urlopen, Request
from urllib.parse import urlparse, urlencode

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


# =============================================================================
# World Bank Indicators Data Access
# =============================================================================

def _download_worldbank_indicator(
    indicator_code: str,
    dest_path: Path,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Dict[str, Any]:
    """Download World Bank indicator data via API.
    
    Args:
        indicator_code: World Bank indicator code (e.g., 'NY.GDP.PCAP.CD')
        dest_path: Destination path for CSV cache file
        start_year: Start year for data (default: all available)
        end_year: End year for data (default: all available)
        
    Returns:
        Dictionary with indicator metadata (name, description, source)
    """
    if dest_path.exists():
        # Return cached metadata if available
        try:
            df = pd.read_csv(dest_path, nrows=0)
            return {
                'name': df.attrs.get('indicator_name', indicator_code),
                'description': df.attrs.get('indicator_description', ''),
                'source': df.attrs.get('indicator_source', '')
            }
        except:
            pass  # Re-download if cache read fails
    
    print(f"Downloading World Bank indicator {indicator_code}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build API URL
    base_url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
    params = {
        'format': 'json',
        'per_page': 20000,  # Large page size to avoid pagination
        'page': 1
    }
    
    if start_year and end_year:
        params['date'] = f"{start_year}:{end_year}"
    
    url = f"{base_url}?{urlencode(params)}"
    
    # Create request with user agent
    req = Request(url, headers={'User-Agent': 'fitkit/1.0 (Python)'})
    
    # Download JSON data
    with urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
    
    # Parse response
    if len(data) < 2 or not isinstance(data[1], list):
        raise ValueError(f"Invalid API response for indicator {indicator_code}")
    
    metadata = data[0]
    records = data[1]
    
    if not records:
        raise ValueError(f"No data found for indicator {indicator_code}")
    
    # Extract indicator metadata from first record
    first_record = records[0]
    indicator_name = first_record.get('indicator', {}).get('value', indicator_code)
    indicator_desc = ''  # API doesn't provide description in data endpoint
    indicator_source = first_record.get('unit', '')
    
    print(f"  Indicator: {indicator_name}")
    print(f"  Records: {len(records)}")
    
    # Convert to DataFrame
    rows = []
    for record in records:
        if record.get('value') is not None:  # Skip null values
            rows.append({
                'country_code': record.get('countryiso3code', ''),
                'country_name': record.get('country', {}).get('value', ''),
                'year': int(record.get('date', 0)),
                'value': float(record.get('value', 0))
            })
    
    if not rows:
        raise ValueError(f"No valid data found for indicator {indicator_code}")
    
    df = pd.DataFrame(rows)
    
    # Remove duplicates (keep last value for each country-year)
    # Duplicates can occur due to data revisions or multiple sources
    df = df.drop_duplicates(subset=['country_code', 'year'], keep='last')
    
    # Pivot to wide format (countries × years)
    df_wide = df.pivot(index='country_code', columns='year', values='value')
    df_wide.index.name = 'country_code'
    
    # Sort columns (years) in ascending order
    df_wide = df_wide[[c for c in sorted(df_wide.columns)]]
    
    # Store metadata as attributes (note: lost when saving to CSV)
    df_wide.attrs['indicator_code'] = indicator_code
    df_wide.attrs['indicator_name'] = indicator_name
    df_wide.attrs['indicator_description'] = indicator_desc
    df_wide.attrs['indicator_source'] = indicator_source
    
    # Save to cache
    df_wide.to_csv(dest_path)
    
    # Also save metadata separately
    metadata_file = dest_path.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'indicator_code': indicator_code,
            'indicator_name': indicator_name,
            'indicator_description': indicator_desc,
            'indicator_source': indicator_source
        }, f, indent=2)
    
    print(f"Downloaded to {dest_path}")
    print(f"  Shape: {df_wide.shape[0]} countries × {df_wide.shape[1]} years")
    
    return {
        'name': indicator_name,
        'description': indicator_desc,
        'source': indicator_source
    }


def load_worldbank_indicator(
    indicator_code: str,
    countries: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    auto_download: bool = True
) -> pd.DataFrame:
    """Load World Bank indicator data.
    
    Downloads and caches data from the World Bank World Development Indicators (WDI)
    database for any indicator. Returns a DataFrame with countries as rows and years
    as columns.
    
    Args:
        indicator_code: World Bank indicator code (e.g., 'NY.GDP.PCAP.CD' for GDP per capita)
        countries: Optional list of ISO3 country codes to filter (e.g., ['USA', 'GBR', 'CHN'])
        start_year: Start year for data (default: all available)
        end_year: End year for data (default: all available)
        auto_download: If True, automatically download data if not cached
        
    Returns:
        DataFrame with country codes as index and years as columns
        - Index: ISO3 country codes (e.g., 'USA', 'GBR', 'CHN')
        - Columns: Years (e.g., 2000, 2001, ..., 2020)
        - Values: Indicator values (NaN for missing data)
        - Attributes: indicator metadata (name, description, source)
        
    Examples:
        >>> from fitkit.datasets import load_worldbank_indicator
        >>> 
        >>> # Load GDP per capita for recent years
        >>> gdp_df = load_worldbank_indicator('NY.GDP.PCAP.CD', start_year=2010, end_year=2020)
        >>> print(f"Shape: {gdp_df.shape}")
        >>> print(f"Countries: {gdp_df.shape[0]}, Years: {gdp_df.shape[1]}")
        >>> 
        >>> # Load Human Capital Index for specific countries
        >>> hci_df = load_worldbank_indicator('HD.HCI.OVRL', countries=['USA', 'CHN', 'IND'])
        >>> print(hci_df)
        >>> 
        >>> # Load any indicator (e.g., Trade as % of GDP)
        >>> trade_df = load_worldbank_indicator('NE.TRD.GNFS.ZS', start_year=2020, end_year=2020)
        >>> print(trade_df['2020'].nlargest(10))  # Most open economies
        
    Common Indicator Codes:
        - NY.GDP.PCAP.CD: GDP per capita, current US$
        - NY.GDP.PCAP.PP.CD: GDP per capita, PPP
        - HD.HCI.OVRL: Human Capital Index
        - NE.TRD.GNFS.ZS: Trade (% of GDP)
        - GB.XPD.RSDV.GD.ZS: R&D expenditure (% of GDP)
        - SP.POP.TOTL: Total population
        - SI.POV.GINI: Gini index
        
    Notes:
        - Data is cached in fitkit/data/worldbank/
        - Some indicators (like HCI) only available for specific years
        - Missing data returned as NaN
        - Country codes use ISO3 standard (USA, CHN, GBR, etc.)
        - **Country code compatibility**: 205 countries overlap with Atlas trade data
        - World Bank includes regional aggregates (AFE, EAS, WLD) - filter with exclude_aggregates
        - For merging with Atlas: both datasets can be joined on ISO3 codes directly
        
    Data Source:
        World Bank World Development Indicators (WDI)
        https://data.worldbank.org/
        
    See Also:
        - load_gdp_per_capita(): Convenience wrapper for GDP per capita
        - load_human_capital_index(): Convenience wrapper for Human Capital Index
        - list_worldbank_indicators(): Browse available indicators
        - list_worldbank_available_countries(): Check country coverage
    """
    data_dir = get_data_dir() / "worldbank"
    
    # Create safe filename from indicator code
    safe_code = indicator_code.replace('.', '_').replace('/', '_')
    cache_file = data_dir / f"{safe_code}.csv"
    metadata_file = data_dir / f"{safe_code}.json"
    
    # Download if needed
    if not cache_file.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"Indicator {indicator_code} not cached. Set auto_download=True to download."
            )
        _download_worldbank_indicator(indicator_code, cache_file, start_year, end_year)
    
    # Load from cache
    df = pd.read_csv(cache_file, index_col='country_code')
    
    # Load metadata if available
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
            df.attrs.update(metadata)
    
    # Convert column names to integers (years)
    df.columns = [int(c) for c in df.columns]
    
    # Filter by countries if specified
    if countries:
        available_countries = [c for c in countries if c in df.index]
        if not available_countries:
            warnings.warn(f"None of the requested countries found in data for {indicator_code}")
            return pd.DataFrame()
        missing_countries = set(countries) - set(available_countries)
        if missing_countries:
            warnings.warn(f"Countries not found in {indicator_code}: {missing_countries}")
        df = df.loc[available_countries]
    
    # Filter by year range if specified
    if start_year or end_year:
        year_cols = df.columns
        if start_year:
            year_cols = [y for y in year_cols if y >= start_year]
        if end_year:
            year_cols = [y for y in year_cols if y <= end_year]
        if not year_cols:
            warnings.warn(f"No data found for year range {start_year}-{end_year}")
            return pd.DataFrame()
        df = df[year_cols]
    
    return df


def load_gdp_per_capita(
    countries: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    auto_download: bool = True
) -> pd.DataFrame:
    """Load GDP per capita data (current US$) from World Bank.
    
    Convenience wrapper for load_worldbank_indicator('NY.GDP.PCAP.CD', ...).
    GDP per capita is the most common metric for economic development and is
    widely used to validate economic fitness scores.
    
    Args:
        countries: Optional list of ISO3 country codes to filter
        start_year: Start year for data (default: all available, typically 1960+)
        end_year: End year for data (default: all available, typically up to 2 years ago)
        auto_download: If True, automatically download data if not cached
        
    Returns:
        DataFrame with country codes as index and years as columns
        - Values are in current US dollars
        - Coverage: 200+ countries, 1960-present (typically 1-2 years lag)
        
    Examples:
        >>> from fitkit import load_gdp_per_capita, load_atlas_trade, fitness_complexity
        >>> 
        >>> # Load GDP for recent period
        >>> gdp_df = load_gdp_per_capita(start_year=2000, end_year=2020)
        >>> 
        >>> # Merge with fitness data
        >>> M, countries, _ = load_atlas_trade(year=2020)
        >>> F, _, _ = fitness_complexity(M)
        >>> countries['fitness'] = F
        >>> 
        >>> comparison = countries.merge(
        ...     gdp_df[[2020]], 
        ...     left_on='country', 
        ...     right_index=True
        ... )
        >>> print(comparison[['country', 'fitness', 2020]].head())
        
    Notes:
        - Uses current US$ (not adjusted for inflation or PPP)
        - For PPP-adjusted GDP, use: load_worldbank_indicator('NY.GDP.PCAP.PP.CD')
        - For constant prices, use: load_worldbank_indicator('NY.GDP.PCAP.KD')
        - Country codes compatible with Atlas data (205 countries overlap)
        - World Bank includes regional aggregates (AFE, EAS, WLD, etc.)
        - For merging with Atlas: use exclude_aggregates=True in list_worldbank_available_countries()
        
    Data Source:
        World Bank World Development Indicators
        https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
    """
    return load_worldbank_indicator(
        'NY.GDP.PCAP.CD',
        countries=countries,
        start_year=start_year,
        end_year=end_year,
        auto_download=auto_download
    )


def load_human_capital_index(
    countries: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    auto_download: bool = True
) -> pd.DataFrame:
    """Load Human Capital Index (HCI) data from World Bank.
    
    Convenience wrapper for load_worldbank_indicator('HD.HCI.OVRL', ...).
    The HCI measures the human capital that a child born today can expect to
    attain by age 18, given the risks of poor health and poor education.
    
    Args:
        countries: Optional list of ISO3 country codes to filter
        start_year: Start year for data (default: all available, typically 2010+)
        end_year: End year for data (default: all available)
        auto_download: If True, automatically download data if not cached
        
    Returns:
        DataFrame with country codes as index and years as columns
        - Values range from 0 to 1 (1 = best possible outcomes)
        - Coverage: 170+ countries, periodic releases (2010, 2012, 2017, 2020)
        - Not all years available (periodic updates, not annual)
        
    Examples:
        >>> from fitkit import load_human_capital_index, load_gdp_per_capita
        >>> 
        >>> # Load HCI for most recent year
        >>> hci_df = load_human_capital_index(start_year=2020, end_year=2020)
        >>> print(f"Countries with HCI data: {len(hci_df)}")
        >>> 
        >>> # Compare HCI with GDP
        >>> gdp_df = load_gdp_per_capita(start_year=2020, end_year=2020)
        >>> comparison = hci_df[[2020]].join(gdp_df[[2020]], lsuffix='_hci', rsuffix='_gdp')
        >>> print(comparison.corr())
        >>> 
        >>> # Analyze fitness-HCI relationship
        >>> from fitkit import load_atlas_trade, fitness_complexity
        >>> M, countries, _ = load_atlas_trade(year=2020)
        >>> F, _, _ = fitness_complexity(M)
        >>> # ... merge with HCI data
        
    Notes:
        - HCI = 0.50 means a child will be only 50% as productive as possible
        - Components: survival, education years, test scores, health
        - Data only available for specific years (not annual)
        - For components, use: 'HD.HCI.EYRS', 'HD.HCI.LAYS', 'HD.HCI.HLOS', 'HD.HCI.MORT'
        - Country codes compatible with Atlas data (205 countries overlap)
        - Smaller coverage than GDP (173 vs 257 countries)
        
    Interpretation:
        - >0.70: High human capital
        - 0.50-0.70: Medium human capital
        - <0.50: Low human capital
        
    Data Source:
        World Bank Human Capital Project
        https://www.worldbank.org/en/publication/human-capital
        https://data.worldbank.org/indicator/HD.HCI.OVRL
    """
    return load_worldbank_indicator(
        'HD.HCI.OVRL',
        countries=countries,
        start_year=start_year,
        end_year=end_year,
        auto_download=auto_download
    )


def list_worldbank_available_countries(
    indicator_code: str,
    auto_download: bool = True,
    exclude_aggregates: bool = False
) -> List[str]:
    """List countries with available data for a specific World Bank indicator.
    
    Args:
        indicator_code: World Bank indicator code (e.g., 'NY.GDP.PCAP.CD')
        auto_download: If True, download indicator data if not cached
        exclude_aggregates: If True, exclude regional/income aggregates (e.g., 'EAS', 'ARB')
        
    Returns:
        List of ISO3 country codes with data for this indicator
        
    Examples:
        >>> from fitkit.datasets import list_worldbank_available_countries
        >>> 
        >>> # Check GDP coverage
        >>> gdp_countries = list_worldbank_available_countries('NY.GDP.PCAP.CD')
        >>> print(f"GDP data available for {len(gdp_countries)} countries")
        >>> 
        >>> # Check HCI coverage (smaller than GDP)
        >>> hci_countries = list_worldbank_available_countries('HD.HCI.OVRL')
        >>> print(f"HCI data available for {len(hci_countries)} countries")
        >>> 
        >>> # Find countries in both datasets (exclude aggregates)
        >>> gdp_real = list_worldbank_available_countries('NY.GDP.PCAP.CD', exclude_aggregates=True)
        >>> print(f"Real countries only: {len(gdp_real)}")
        
    Notes:
        - World Bank data includes regional aggregates (e.g., 'EAS' for East Asia & Pacific)
        - These aggregates won't match with Atlas trade data
        - Use exclude_aggregates=True when merging with Atlas data
        - Aggregate codes: 3-letter codes that don't represent individual countries
          (e.g., 'AFE', 'ARB', 'EAS', 'EUU', 'LCN', 'SSF', 'WLD')
    """
    df = load_worldbank_indicator(indicator_code, auto_download=auto_download)
    # Filter out NaN/empty indices
    countries = [c for c in df.index if c and isinstance(c, str) and not pd.isna(c)]
    
    if exclude_aggregates:
        # Common World Bank aggregate codes
        # Regional: AFE, AFW, ARB, CEB, EAP, EAS, ECA, ECS, EMU, EUU, FCS, HPC, LAC, LCN, MEA, MNA, NAC, OED, OSS, PRE, PSS, PST, SAS, SSA, SSF, TEA, TEC, TLA, TMN, TSA, TSS, WLD
        # Income: HIC, IBD, IBT, IDA, IDB, IDX, INX, LDC, LIC, LMC, LMY, LTE, MIC, NOC, OEC, UMC
        aggregates = {
            'AFE', 'AFW', 'ARB', 'CEB', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 
            'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN',
            'LDC', 'LIE', 'LIC', 'LMC', 'LMY', 'LTE', 'MAF', 'MCO', 'MEA', 'MIC', 'MNA', 
            'NAC', 'NOC', 'OEC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 
            'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'
        }
        countries = [c for c in countries if c not in aggregates]
    
    return sorted(countries)
