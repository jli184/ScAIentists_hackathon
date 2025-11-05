"""
Enrich a single city CSV file with climate data.
Only processes rows with missing climate data.

Usage: python enrich_single_city.py <csv_file_path>
"""
import pandas as pd
import numpy as np
import requests
import sys
import time
from pathlib import Path

def fetch_climate_features(lat, lon, year):
    """
    Fetch climate features for a location-year

    Returns dict with:
    - spring_temp: Jan-March average temperature (°C)
    - spring_gdd: Growing Degree Days Jan-March (base 5°C)
    - winter_chill_days: Dec(year-1) to Feb(year) days below 7°C
    - spring_precip: Jan-March total precipitation (mm)
    """
    # Open-Meteo historical API only goes back to 1940
    if year < 1940:
        return {
            'spring_temp': np.nan,
            'spring_gdd': np.nan,
            'winter_chill_days': np.nan,
            'spring_precip': np.nan
        }

    try:
        # Fetch winter period (Dec previous year to Feb current year)
        winter_start = f"{year-1}-12-01"
        winter_end = f"{year}-02-28"

        url = "https://archive-api.open-meteo.com/v1/archive"
        params_winter = {
            "latitude": lat,
            "longitude": lon,
            "start_date": winter_start,
            "end_date": winter_end,
            "daily": "temperature_2m_mean",
            "timezone": "auto"
        }

        response_winter = requests.get(url, params=params_winter, timeout=30)

        if response_winter.status_code != 200:
            winter_chill_days = np.nan
        else:
            data_winter = response_winter.json()
            if 'daily' in data_winter and 'temperature_2m_mean' in data_winter['daily']:
                temps_winter = [t for t in data_winter['daily']['temperature_2m_mean'] if t is not None]
                winter_chill_days = sum(1 for t in temps_winter if t < 7.0)
            else:
                winter_chill_days = np.nan

        # Small delay to respect API rate limits
        time.sleep(0.05)

        # Fetch spring period (Jan-March current year)
        spring_start = f"{year}-01-01"
        spring_end = f"{year}-03-31"

        params_spring = {
            "latitude": lat,
            "longitude": lon,
            "start_date": spring_start,
            "end_date": spring_end,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "auto"
        }

        response_spring = requests.get(url, params=params_spring, timeout=30)

        if response_spring.status_code != 200:
            return {
                'spring_temp': np.nan,
                'spring_gdd': np.nan,
                'winter_chill_days': winter_chill_days,
                'spring_precip': np.nan
            }

        data_spring = response_spring.json()

        if 'daily' not in data_spring:
            return {
                'spring_temp': np.nan,
                'spring_gdd': np.nan,
                'winter_chill_days': winter_chill_days,
                'spring_precip': np.nan
            }

        # Extract spring data
        temps_spring = [t for t in data_spring['daily']['temperature_2m_mean'] if t is not None]
        precip_spring = [p for p in data_spring['daily']['precipitation_sum'] if p is not None]

        if not temps_spring:
            return {
                'spring_temp': np.nan,
                'spring_gdd': np.nan,
                'winter_chill_days': winter_chill_days,
                'spring_precip': np.nan
            }

        # Calculate features
        spring_temp = sum(temps_spring) / len(temps_spring)
        spring_gdd = sum(max(0, t - 5.0) for t in temps_spring)  # Base 5°C
        spring_precip = sum(precip_spring) if precip_spring else np.nan

        return {
            'spring_temp': round(spring_temp, 2),
            'spring_gdd': round(spring_gdd, 2),
            'winter_chill_days': winter_chill_days,
            'spring_precip': round(spring_precip, 2) if spring_precip is not np.nan else np.nan
        }

    except Exception as e:
        return {
            'spring_temp': np.nan,
            'spring_gdd': np.nan,
            'winter_chill_days': np.nan,
            'spring_precip': np.nan
        }

def check_if_needs_enrichment(df):
    """Check if CSV needs enrichment (has missing climate data)"""
    # Check if climate columns exist
    climate_cols = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']

    if not all(col in df.columns for col in climate_cols):
        return True, len(df[df['year'] >= 1940])

    # Check for rows with missing data (year >= 1940)
    enrichable_rows = df[df['year'] >= 1940]
    if len(enrichable_rows) == 0:
        return False, 0

    # Count rows with any missing climate data
    missing_mask = enrichable_rows[climate_cols].isna().any(axis=1)
    missing_count = missing_mask.sum()

    return missing_count > 0, missing_count

def enrich_city_csv(csv_path):
    """Enrich a single city CSV file with climate data"""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        return False

    print(f"Processing: {csv_path.name}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ERROR: Could not read CSV: {e}")
        return False

    # Check if needs enrichment
    needs_enrichment, missing_count = check_if_needs_enrichment(df)

    if not needs_enrichment:
        print(f"  ✓ Already complete (no missing data)")
        return True

    print(f"  Found {missing_count} rows needing enrichment")

    # Add columns if they don't exist
    for col in ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']:
        if col not in df.columns:
            df[col] = np.nan

    # Get location coordinates (should be same for all rows in a city file)
    if len(df) == 0:
        print(f"  ERROR: Empty CSV file")
        return False

    lat = df['lat'].iloc[0]
    lon = df['long'].iloc[0]
    location_name = df['location'].iloc[0]

    # Process rows that need enrichment
    enriched_count = 0
    skipped_count = 0

    for idx, row in df.iterrows():
        year = int(row['year'])

        # Skip if year < 1940 (no data available)
        if year < 1940:
            continue

        # Skip if already has complete data
        if pd.notna(row['spring_temp']) and pd.notna(row['spring_gdd']) and \
           pd.notna(row['winter_chill_days']) and pd.notna(row['spring_precip']):
            skipped_count += 1
            continue

        # Fetch climate data
        climate = fetch_climate_features(lat, lon, year)

        # Update dataframe
        df.loc[idx, 'spring_temp'] = climate['spring_temp']
        df.loc[idx, 'spring_gdd'] = climate['spring_gdd']
        df.loc[idx, 'winter_chill_days'] = climate['winter_chill_days']
        df.loc[idx, 'spring_precip'] = climate['spring_precip']

        enriched_count += 1

        # Progress indicator
        if enriched_count % 5 == 0:
            print(f"    Progress: {enriched_count}/{missing_count} rows enriched", end='\r')

    # Save enriched CSV
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Enriched {enriched_count} rows (skipped {skipped_count} complete rows)")

    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python enrich_single_city.py <csv_file_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    success = enrich_city_csv(csv_path)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
