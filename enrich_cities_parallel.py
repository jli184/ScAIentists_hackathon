"""
Parallel enrichment of city CSV files with climate data.
Scans all city CSV files, identifies those with missing data, and processes them in parallel.

Usage:
    python enrich_cities_parallel.py [--workers N] [--batch-size N]

Options:
    --workers N      Number of parallel workers (default: 4)
    --batch-size N   Number of files to process per batch (default: 10)
"""
import pandas as pd
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

def check_file_needs_enrichment(csv_path):
    """Check if a CSV file needs enrichment and return (path, needs_enrichment, missing_count)"""
    try:
        df = pd.read_csv(csv_path)

        # Check if climate columns exist
        climate_cols = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']

        if not all(col in df.columns for col in climate_cols):
            enrichable = len(df[df['year'] >= 1940])
            return (str(csv_path), True, enrichable, len(df))

        # Check for rows with missing data (year >= 1940)
        enrichable_rows = df[df['year'] >= 1940]
        if len(enrichable_rows) == 0:
            return (str(csv_path), False, 0, len(df))

        # Count rows with any missing climate data
        missing_mask = enrichable_rows[climate_cols].isna().any(axis=1)
        missing_count = missing_mask.sum()

        return (str(csv_path), missing_count > 0, missing_count, len(df))

    except Exception as e:
        print(f"Warning: Could not check {csv_path.name}: {e}")
        return (str(csv_path), False, 0, 0)

def enrich_file_subprocess(csv_path):
    """Enrich a single file using subprocess"""
    try:
        result = subprocess.run(
            [sys.executable, "enrich_single_city.py", csv_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per file
        )
        return (csv_path, result.returncode == 0, result.stdout)
    except subprocess.TimeoutExpired:
        return (csv_path, False, "ERROR: Timeout after 5 minutes")
    except Exception as e:
        return (csv_path, False, f"ERROR: {e}")

def scan_city_files(city_dir="data/by_city"):
    """Scan all city CSV files and identify which need enrichment"""
    city_dir = Path(city_dir)

    if not city_dir.exists():
        print(f"ERROR: Directory not found: {city_dir}")
        return []

    csv_files = list(city_dir.glob("*.csv"))
    print(f"Scanning {len(csv_files)} city CSV files...")
    print()

    # Check each file
    files_to_enrich = []
    complete_files = []

    for csv_file in csv_files:
        path, needs_enrichment, missing_count, total_rows = check_file_needs_enrichment(csv_file)

        if needs_enrichment:
            files_to_enrich.append((path, missing_count, total_rows))
        else:
            complete_files.append(path)

    return files_to_enrich, complete_files

def main():
    parser = argparse.ArgumentParser(description="Parallel enrichment of city CSV files")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=10, help="Files per batch")
    parser.add_argument("--dir", type=str, default="data/by_city", help="Directory containing city CSV files")
    args = parser.parse_args()

    print("="*70)
    print("Parallel City CSV Enrichment")
    print("="*70)
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Scan for files needing enrichment
    files_to_enrich, complete_files = scan_city_files(args.dir)

    print(f"Summary:")
    print(f"  ✓ Already complete: {len(complete_files)} files")
    print(f"  ⚠ Need enrichment: {len(files_to_enrich)} files")

    if len(files_to_enrich) == 0:
        print()
        print("="*70)
        print("All files are already complete!")
        print("="*70)
        return

    # Show top files by missing count
    files_to_enrich.sort(key=lambda x: x[1], reverse=True)
    print()
    print(f"Top 10 files by missing rows:")
    for path, missing, total in files_to_enrich[:10]:
        filename = Path(path).name
        print(f"  {filename}: {missing}/{total} rows missing")

    print()
    print("="*70)
    print(f"Starting parallel enrichment of {len(files_to_enrich)} files...")
    print("="*70)
    print()

    # Process in batches
    total_files = len(files_to_enrich)
    completed = 0
    failed = []

    for batch_start in range(0, total_files, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total_files)
        batch = files_to_enrich[batch_start:batch_end]
        batch_num = (batch_start // args.batch_size) + 1
        total_batches = (total_files + args.batch_size - 1) // args.batch_size

        print(f"Batch {batch_num}/{total_batches}: Processing {len(batch)} files...")
        batch_start_time = time.time()

        # Process batch in parallel
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(enrich_file_subprocess, path): path
                      for path, _, _ in batch}

            for future in as_completed(futures):
                csv_path, success, output = future.result()
                filename = Path(csv_path).name

                if success:
                    # Extract enriched count from output
                    lines = output.strip().split('\n')
                    summary_line = [l for l in lines if '✓ Enriched' in l or '✓ Already complete' in l]
                    if summary_line:
                        print(f"  ✓ {filename}: {summary_line[-1].strip()}")
                    else:
                        print(f"  ✓ {filename}")
                    completed += 1
                else:
                    print(f"  ✗ {filename}: FAILED")
                    if output:
                        print(f"    {output}")
                    failed.append(csv_path)

        batch_time = time.time() - batch_start_time
        print(f"  Batch completed in {batch_time:.1f}s")
        print()

    # Summary
    print("="*70)
    print("Enrichment Complete!")
    print("="*70)
    print(f"Successfully enriched: {completed}/{total_files} files")

    if failed:
        print(f"Failed: {len(failed)} files")
        print()
        print("Failed files:")
        for path in failed:
            print(f"  - {Path(path).name}")

    print("="*70)

if __name__ == "__main__":
    main()
