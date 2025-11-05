"""
Optimized script to enrich CSVs with climate data
"""
import pandas as pd
import numpy as np
import requests
import glob
import time
from collections import defaultdict

def fetch_climate_for_year(lat, lon, year):
    """Fetch climate features for one location-year"""
    if year < 1940:
        return None

    try:
        url = "https://archive-api.open-meteo.com/v1/archive"

        # Fetch Dec-Feb for chilling
        params_winter = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year-1}-12-01",
            "end_date": f"{year}-02-28",
            "daily": "temperature_2m_mean",
            "timezone": "auto"
        }
        resp_winter = requests.get(url, params=params_winter, timeout=30)

        if resp_winter.status_code == 200:
            data_winter = resp_winter.json()
            temps_winter = [t for t in data_winter['daily']['temperature_2m_mean'] if t is not None]
            winter_chill = sum(1 for t in temps_winter if t < 7.0)
        else:
            winter_chill = np.nan

        time.sleep(0.05)  # Rate limiting

        # Fetch Jan-March for spring conditions
        params_spring = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-03-31",
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "auto"
        }
        resp_spring = requests.get(url, params=params_spring, timeout=30)

        if resp_spring.status_code != 200:
            return {'winter_chill_days': winter_chill, 'spring_temp': np.nan,
                    'spring_gdd': np.nan, 'spring_precip': np.nan}

        data_spring = resp_spring.json()
        temps = [t for t in data_spring['daily']['temperature_2m_mean'] if t is not None]
        precip = [p for p in data_spring['daily']['precipitation_sum'] if p is not None]

        if not temps:
            return {'winter_chill_days': winter_chill, 'spring_temp': np.nan,
                    'spring_gdd': np.nan, 'spring_precip': np.nan}

        spring_temp = sum(temps) / len(temps)
        spring_gdd = sum(max(0, t - 5.0) for t in temps)
        spring_precip = sum(precip) if precip else np.nan

        return {
            'spring_temp': round(spring_temp, 2),
            'spring_gdd': round(spring_gdd, 2),
            'winter_chill_days': winter_chill,
            'spring_precip': round(spring_precip, 2) if not np.isnan(spring_precip) else np.nan
        }

    except Exception as e:
        print(f"      Error for year {year}: {e}")
        return None

def enrich_csv_file(csv_path):
    """Enrich one CSV file"""
    filename = csv_path.split('/')[-1]
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)

    print(f"Total records: {len(df)}")
    records_1940plus = len(df[df['year'] >= 1940])
    print(f"Records >= 1940 (can be enriched): {records_1940plus}")

    # Add new columns if they don't exist
    if 'spring_temp' not in df.columns:
        df['spring_temp'] = np.nan
        df['spring_gdd'] = np.nan
        df['winter_chill_days'] = np.nan
        df['spring_precip'] = np.nan

    # Check which locations are already enriched
    locations_to_process = []
    locations_completed = []

    for location_name in df['location'].unique():
        location_df = df[df['location'] == location_name]
        enrichable = location_df[location_df['year'] >= 1940]

        # Check if this location has any enriched records
        has_climate = location_df['spring_temp'].notna().sum()

        if has_climate > 0:
            locations_completed.append(location_name)
        else:
            locations_to_process.append(location_name)

    if locations_completed:
        print(f"\n✓ Already enriched: {len(locations_completed)} locations")

    if not locations_to_process:
        print(f"All locations already enriched, skipping file!")
        return

    print(f"📋 To process: {len(locations_to_process)} locations remaining")

    # Process remaining locations
    for location_name in locations_to_process:
        location_df = df[df['location'] == location_name]
        lat = location_df['lat'].iloc[0]
        lon = location_df['long'].iloc[0]

        enrichable = location_df[location_df['year'] >= 1940]
        print(f"\n  {location_name}: {len(enrichable)} enrichable records")

        for i, (idx, row) in enumerate(enrichable.iterrows(), 1):
            year = int(row['year'])

            climate = fetch_climate_for_year(lat, lon, year)

            if climate:
                df.loc[idx, 'spring_temp'] = climate['spring_temp']
                df.loc[idx, 'spring_gdd'] = climate['spring_gdd']
                df.loc[idx, 'winter_chill_days'] = climate['winter_chill_days']
                df.loc[idx, 'spring_precip'] = climate['spring_precip']

            # Progress
            if i % 5 == 0 or i == len(enrichable):
                print(f"    Progress: {i}/{len(enrichable)}", end='\r', flush=True)

            time.sleep(0.05)  # Rate limiting

        print(f"    Progress: {len(enrichable)}/{len(enrichable)} ✓")

        # Save after each location to preserve progress
        df.to_csv(csv_path, index=False)
        print(f"    Saved progress for {location_name}")

    # Final save confirmation
    print(f"\n✓ All locations in {csv_path} enriched and saved")

def main():
    print("="*60)
    print("Enriching Cherry Blossom CSVs with Climate Data")
    print("="*60)

    all_csv_files = sorted(glob.glob("data/*.csv"))

    # Prioritize non-Japan files first, then Japan
    priority_order = []
    japan_files = []

    for csv_file in all_csv_files:
        if 'japan.csv' in csv_file:
            japan_files.append(csv_file)
        elif 'toronto.csv' not in csv_file:  # Skip Toronto (already done)
            priority_order.append(csv_file)

    # Put Japan at the end
    csv_files = priority_order + japan_files

    print(f"\nFound {len(csv_files)} CSV files to process")
    print(f"Priority order: Other datasets first, then Japan")

    # Show summary
    total_records = 0
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        total_records += len(df)

        # Check enrichment status
        if 'spring_temp' in df.columns:
            enriched = df['spring_temp'].notna().sum()
            pct = enriched/len(df)*100 if len(df) > 0 else 0
            status = f"({enriched}/{len(df)} = {pct:.0f}% enriched)"
        else:
            status = "(not started)"

        print(f"  {csv_file.split('/')[-1]:20} {len(df):5} records {status}")

    print(f"\nTotal records across all files: {total_records}")

    print("\n🎯 Strategy: Enrich other datasets first for diversity, then complete Japan")
    print("Starting enrichment (saves after each location)...")

    start_time = time.time()

    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n\n[File {i}/{len(csv_files)}]")
        try:
            enrich_csv_file(csv_file)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ All files enriched in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
