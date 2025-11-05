"""
Enrich all cherry blossom CSV files with year-specific climate data
Adds: spring_temp, spring_gdd, winter_chill_days, spring_precip
"""
import pandas as pd
import numpy as np
import requests
import glob
import time
from datetime import datetime

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
            print(f"    Warning: Failed to fetch winter data for {year}")
            winter_chill_days = np.nan
        else:
            data_winter = response_winter.json()
            if 'daily' in data_winter and 'temperature_2m_mean' in data_winter['daily']:
                temps_winter = [t for t in data_winter['daily']['temperature_2m_mean'] if t is not None]
                winter_chill_days = sum(1 for t in temps_winter if t < 7.0)
            else:
                winter_chill_days = np.nan

        # Small delay to respect API rate limits
        time.sleep(0.1)

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
            print(f"    Warning: Failed to fetch spring data for {year}")
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
        print(f"    Error fetching data for {year}: {e}")
        return {
            'spring_temp': np.nan,
            'spring_gdd': np.nan,
            'winter_chill_days': np.nan,
            'spring_precip': np.nan
        }

def enrich_csv(csv_path):
    """Enrich a single CSV file with climate data"""
    print(f"\nProcessing: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check if already enriched
    if 'spring_temp' in df.columns:
        print(f"  Already enriched, skipping...")
        return

    print(f"  {len(df)} records to enrich")

    # Add new columns
    df['spring_temp'] = np.nan
    df['spring_gdd'] = np.nan
    df['winter_chill_days'] = np.nan
    df['spring_precip'] = np.nan

    # Get unique location coordinates
    location_coords = df.groupby('location').first()[['lat', 'long']].to_dict('index')

    # Process each location's years together to optimize API calls
    for location_name, coords in location_coords.items():
        location_df = df[df['location'] == location_name]
        lat = coords['lat']
        lon = coords['long']

        print(f"  {location_name}: {len(location_df)} records")

        for idx, row in location_df.iterrows():
            year = int(row['year'])

            # Fetch climate data
            climate = fetch_climate_features(lat, lon, year)

            # Update dataframe
            df.loc[idx, 'spring_temp'] = climate['spring_temp']
            df.loc[idx, 'spring_gdd'] = climate['spring_gdd']
            df.loc[idx, 'winter_chill_days'] = climate['winter_chill_days']
            df.loc[idx, 'spring_precip'] = climate['spring_precip']

            # Progress indicator
            if (idx % 10) == 0:
                print(f"    Processed {idx - location_df.index[0] + 1}/{len(location_df)} years", end='\r')

        print(f"    Completed {location_name}                    ")

    # Save enriched CSV
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved enriched data to {csv_path}")

def main():
    print("="*60)
    print("Enriching Cherry Blossom CSVs with Climate Data")
    print("="*60)
    print("\nFeatures to add:")
    print("  - spring_temp: Jan-March average temperature (°C)")
    print("  - spring_gdd: Growing Degree Days Jan-March (base 5°C)")
    print("  - winter_chill_days: Dec-Feb days below 7°C")
    print("  - spring_precip: Jan-March total precipitation (mm)")
    print("\nNote: Only years >= 1940 have weather data")

    # Find all CSV files
    csv_files = glob.glob("data/*.csv")
    print(f"\nFound {len(csv_files)} CSV files to process")

    # Process each file
    for csv_file in csv_files:
        try:
            enrich_csv(csv_file)
        except Exception as e:
            print(f"  ERROR processing {csv_file}: {e}")
            continue

    print("\n" + "="*60)
    print("✓ Climate enrichment complete!")
    print("="*60)

if __name__ == "__main__":
    main()
