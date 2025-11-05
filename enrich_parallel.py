"""
Parallel enrichment - process multiple CSV files simultaneously
"""
import subprocess
import time

# Files to enrich (skip toronto - already done)
files_to_process = [
    'data/japan.csv',
    'data/kyoto.csv',
    'data/liestal.csv',
    'data/meteoswiss.csv',
    'data/nyc.csv',
    'data/south_korea.csv',
    'data/vancouver.csv',
    'data/washingtondc.csv'
]

# Process 3 files at a time to avoid overwhelming API
batch_size = 3

print("="*60)
print("Parallel CSV Enrichment")
print("="*60)
print(f"Processing {len(files_to_process)} files in batches of {batch_size}")
print()

def enrich_single_file(csv_path):
    """Enrich a single CSV file"""
    import pandas as pd
    import numpy as np
    import requests
    import time

    # Same fetch function as before
    def fetch_climate_for_year(lat, lon, year):
        if year < 1940:
            return None
        try:
            url = "https://archive-api.open-meteo.com/v1/archive"

            # Winter chilling
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

            time.sleep(0.05)

            # Spring conditions
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
            return None

    # Load and enrich
    print(f"Starting: {csv_path}")
    df = pd.read_csv(csv_path)

    # Check if already enriched
    if 'spring_temp' in df.columns:
        print(f"{csv_path}: Already enriched")
        return

    # Add columns
    df['spring_temp'] = np.nan
    df['spring_gdd'] = np.nan
    df['winter_chill_days'] = np.nan
    df['spring_precip'] = np.nan

    # Process each location
    for location_name in df['location'].unique():
        location_df = df[df['location'] == location_name]
        lat = location_df['lat'].iloc[0]
        lon = location_df['long'].iloc[0]

        enrichable = location_df[location_df['year'] >= 1940]
        if len(enrichable) == 0:
            continue

        print(f"  {csv_path} - {location_name}: {len(enrichable)} records")

        for idx, row in enrichable.iterrows():
            year = int(row['year'])
            climate = fetch_climate_for_year(lat, lon, year)

            if climate:
                df.loc[idx, 'spring_temp'] = climate['spring_temp']
                df.loc[idx, 'spring_gdd'] = climate['spring_gdd']
                df.loc[idx, 'winter_chill_days'] = climate['winter_chill_days']
                df.loc[idx, 'spring_precip'] = climate['spring_precip']

            time.sleep(0.05)

        # Save after each location
        df.to_csv(csv_path, index=False)

    print(f"✓ Completed: {csv_path}")

# Process in batches
for i in range(0, len(files_to_process), batch_size):
    batch = files_to_process[i:i+batch_size]

    print(f"\nBatch {i//batch_size + 1}: Processing {len(batch)} files")
    for f in batch:
        print(f"  - {f}")

    # Start all processes in batch
    processes = []
    for csv_file in batch:
        cmd = f"python -c \"exec(open('enrich_parallel.py').read().split('# START')[1]); enrich_single_file('{csv_file}')\""
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((csv_file, p))

    # Wait for all to complete
    for csv_file, p in processes:
        p.wait()
        print(f"  ✓ {csv_file} done")

    print(f"Batch {i//batch_size + 1} complete!")

print("\n" + "="*60)
print("All files enriched!")
print("="*60)
