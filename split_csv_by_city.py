"""
Split large CSV files by city to enable efficient parallel processing.

Each city gets its own CSV file with all rows for that location.
"""
import csv
import os
from collections import defaultdict

def split_csv_by_city(input_file, output_dir="data/by_city"):
    """
    Split a CSV file by city (location column).

    Args:
        input_file: Path to input CSV file
        output_dir: Directory to store split files (default: data/by_city)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the input file and group by location
    city_data = defaultdict(list)
    header = None

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames

        for row in reader:
            location = row['location']
            city_data[location].append(row)

    print(f"Found {len(city_data)} unique cities")

    # Write each city to its own file
    base_name = os.path.basename(input_file).replace('.csv', '')

    for location, rows in city_data.items():
        # Create a safe filename from the location
        # Replace "/" with "_" and remove any other problematic characters
        safe_location = location.replace('/', '_').replace(' ', '_')
        safe_location = ''.join(c for c in safe_location if c.isalnum() or c in ('_', '-'))

        output_file = os.path.join(output_dir, f"{base_name}_{safe_location}.csv")

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  {location}: {len(rows)} rows -> {output_file}")

    print(f"✓ Split complete: {len(city_data)} files created")
    return len(city_data)

def main():
    """Split all large CSV files by city"""
    files_to_split = [
        'data/japan.csv',
        'data/meteoswiss.csv',
        'data/south_korea.csv'
    ]

    print("="*60)
    print("Splitting CSV files by city")
    print("="*60)
    print()

    total_files = 0
    for csv_file in files_to_split:
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping")
            continue

        count = split_csv_by_city(csv_file)
        total_files += count
        print()

    print("="*60)
    print(f"Complete! Created {total_files} city-specific CSV files")
    print("Files are in: data/by_city/")
    print("="*60)

if __name__ == '__main__':
    main()
