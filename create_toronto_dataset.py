"""
Create Toronto cherry blossom dataset from historical High Park bloom data
"""
import pandas as pd
from datetime import datetime

# Toronto High Park coordinates
TORONTO_LAT = 43.646548
TORONTO_LONG = -79.463690
TORONTO_ALT = 106.277  # meters

# Historical peak bloom data from Sakura in High Park
# Format: (year, start_date, end_date)
historical_data = [
    (2012, "2012-04-10", "2012-04-17"),
    (2013, "2013-04-30", "2013-05-06"),
    (2014, "2014-05-12", "2014-05-21"),
    (2015, "2015-05-05", "2015-05-10"),
    (2016, "2016-05-07", "2016-05-12"),  # Note: only 25% bloom
    (2017, "2017-04-24", "2017-05-02"),
    (2018, "2018-05-07", "2018-05-12"),
    (2019, "2019-05-10", "2019-05-17"),
    (2020, "2020-05-03", "2020-05-09"),
    (2021, "2021-04-20", "2021-04-28"),
    (2022, "2022-05-05", "2022-05-12"),
    (2023, "2023-04-20", "2023-04-28"),
    (2024, "2024-04-20", "2024-04-28"),
    (2025, "2025-05-03", "2025-05-09"),
]

# Calculate midpoint bloom date and day of year
rows = []
for year, start_str, end_str in historical_data:
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")

    # Use midpoint of bloom period
    diff_days = (end_date - start_date).days
    midpoint = start_date + pd.Timedelta(days=diff_days // 2)

    # Calculate day of year
    bloom_doy = midpoint.timetuple().tm_yday
    bloom_date = midpoint.strftime("%Y-%m-%d")

    rows.append({
        'location': 'toronto',
        'lat': TORONTO_LAT,
        'long': TORONTO_LONG,
        'alt': TORONTO_ALT,
        'year': year,
        'bloom_date': bloom_date,
        'bloom_doy': bloom_doy
    })

# Create DataFrame
df = pd.DataFrame(rows)

# Save to CSV
output_path = "data/toronto.csv"
df.to_csv(output_path, index=False)

print(f"Created Toronto dataset: {output_path}")
print(f"Records: {len(df)}")
print(f"Years: {df['year'].min()} - {df['year'].max()}")
print(f"Bloom DOY range: {df['bloom_doy'].min()} - {df['bloom_doy'].max()}")
print("\nFirst few rows:")
print(df.head())
print("\nFull dataset:")
print(df.to_string(index=False))
