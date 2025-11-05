import pandas as pd
import os

data_dir = "data"

# Get all CSV files
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

print("=" * 80)
print("CHERRY BLOSSOM DATA EXPLORATION")
print("=" * 80)

for csv_file in sorted(csv_files):
    filepath = os.path.join(data_dir, csv_file)
    df = pd.read_csv(filepath)

    print(f"\n📁 {csv_file}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df['year'].min()} - {df['year'].max()}")

    if 'bloom_doy' in df.columns:
        print(f"   Bloom DOY range: {df['bloom_doy'].min()} - {df['bloom_doy'].max()}")

    # Show first few rows
    print(f"\n   Sample data:")
    print(df.head(3).to_string(index=False))
    print()

# Combined statistics
print("\n" + "=" * 80)
print("COMBINED DATASET SUMMARY")
print("=" * 80)

all_dfs = []
for csv_file in csv_files:
    filepath = os.path.join(data_dir, csv_file)
    df = pd.read_csv(filepath)
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal records: {len(combined_df)}")
print(f"Year range: {combined_df['year'].min()} - {combined_df['year'].max()}")
print(f"Unique locations: {combined_df['location'].nunique()}")
print(f"\nColumn data types:")
print(combined_df.dtypes)
print(f"\nMissing values:")
print(combined_df.isnull().sum())
print(f"\nBloom DOY statistics:")
print(combined_df['bloom_doy'].describe())
