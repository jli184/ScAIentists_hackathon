"""
Utility functions for loading cherry blossom data from various sources.

This module provides a centralized way to load data from both:
- Original country-level CSV files in data/
- Individual city CSV files in data/by_city/
"""
import pandas as pd
import glob
from pathlib import Path

def load_all_data(include_city_files=True, data_dir="data"):
    """
    Load all cherry blossom datasets from both country and city files.

    Args:
        include_city_files: If True, also load individual city CSV files
        data_dir: Base data directory (default: "data")

    Returns:
        pandas.DataFrame with combined data from all sources
    """
    data_dir = Path(data_dir)
    all_dfs = []

    # Load original country-level files
    country_files = list(data_dir.glob("*.csv"))
    print(f"Loading data from {len(country_files)} country-level files...")

    for file in country_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {file.name}: {e}")

    # Load individual city files if requested
    if include_city_files:
        city_dir = data_dir / "by_city"
        if city_dir.exists():
            city_files = list(city_dir.glob("*.csv"))
            print(f"Loading data from {len(city_files)} city-level files...")

            for file in city_files:
                try:
                    df = pd.read_csv(file)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"  Warning: Could not load {file.name}: {e}")

    if not all_dfs:
        raise ValueError("No data files could be loaded!")

    # Combine all dataframes
    combined = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates (in case same data is in both country and city files)
    initial_len = len(combined)
    combined = combined.drop_duplicates(subset=['location', 'year', 'bloom_date'], keep='first')
    duplicates_removed = initial_len - len(combined)

    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate records")

    print(f"\nLoaded {len(combined)} total records")
    print(f"  Unique locations: {combined['location'].nunique()}")
    print(f"  Year range: {combined['year'].min()}-{combined['year'].max()}")

    # Check for climate data completeness
    climate_cols = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']
    if all(col in combined.columns for col in climate_cols):
        enrichable = combined[combined['year'] >= 1940]
        with_climate = enrichable[climate_cols].notna().all(axis=1).sum()
        total_enrichable = len(enrichable)
        pct = (with_climate / total_enrichable * 100) if total_enrichable > 0 else 0
        print(f"  Climate data: {with_climate}/{total_enrichable} enrichable records ({pct:.1f}%)")

    return combined

def load_city_files_only(city_dir="data/by_city"):
    """
    Load only the individual city CSV files.

    Args:
        city_dir: Directory containing city CSV files

    Returns:
        pandas.DataFrame with combined city data
    """
    city_dir = Path(city_dir)

    if not city_dir.exists():
        raise ValueError(f"City directory not found: {city_dir}")

    city_files = list(city_dir.glob("*.csv"))
    print(f"Loading {len(city_files)} city files from {city_dir}...")

    all_dfs = []
    for file in city_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {file.name}: {e}")

    if not all_dfs:
        raise ValueError("No city files could be loaded!")

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"Loaded {len(combined)} total records")
    print(f"  Unique locations: {combined['location'].nunique()}")
    print(f"  Year range: {combined['year'].min()}-{combined['year'].max()}")

    return combined

def get_data_summary(df):
    """
    Print a summary of the loaded data.

    Args:
        df: pandas DataFrame with cherry blossom data
    """
    print("\n" + "="*60)
    print("Data Summary")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Unique locations: {df['location'].nunique()}")
    print(f"Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"\nColumns: {', '.join(df.columns)}")

    # Climate data status
    climate_cols = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']
    if all(col in df.columns for col in climate_cols):
        print(f"\nClimate Data Coverage:")
        for col in climate_cols:
            non_null = df[col].notna().sum()
            pct = (non_null / len(df) * 100)
            print(f"  {col}: {non_null}/{len(df)} ({pct:.1f}%)")

    print("="*60)
