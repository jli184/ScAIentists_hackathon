# Climate Features Enhancement

## Overview

All prediction scripts have been updated to automatically detect and use climate features when available in the data.

## Climate Features Added

1. **spring_temp**: Jan-March average temperature (°C)
   - Warmer springs → earlier blooms

2. **spring_gdd**: Growing Degree Days (base 5°C)
   - Accumulated warmth needed for bloom

3. **winter_chill_days**: Dec-Feb days below 7°C
   - Chilling requirement for dormancy break

4. **spring_precip**: Jan-March total precipitation (mm)
   - Moisture availability

## Data Source

- **Open-Meteo Historical Weather API**
- Available for years >= 1940
- Data fetched per location-year

## Enrichment Scripts

### `enrich_csvs_optimized.py`
- Currently running (in background)
- Enriches all CSVs with climate data
- Progress: ~9/109 Japanese locations completed
- ETA: ~30-60 minutes total

### Toronto Sample Data
Already enriched with climate features. Example:
```
Year: 2012, Spring Temp: 1.4°C, Bloom DOY: 104 (early)
Year: 2014, Spring Temp: -6.5°C, Bloom DOY: 136 (late)
```

## Updated Prediction Scripts

All scripts now **auto-detect** climate features:

### 1. `tabpfn_cherry_blossom_prediction.py`
**Main prediction pipeline**
- Automatically uses climate features if available in CSVs
- Falls back to base features (lat/long/alt/year) if not
- Prints feature list used
- Drops records with missing climate data

**Usage:**
```bash
python tabpfn_cherry_blossom_prediction.py
```

**Output:**
- Toronto predictions with all available features
- Benchmark on other locations (80/20 split)
- Saves: `toronto_predictions.csv`

### 2. `tabpfn_simple_prediction.py`
**Simple/Ensemble approach**
- Single model (1000 samples) or ensemble (10 models)
- Auto-detects climate features
- Interactive prompt for ensemble

**Usage:**
```bash
python tabpfn_simple_prediction.py
```

**Output:**
- Single model predictions
- Optional ensemble predictions
- Saves: `toronto_predictions_simple.csv`, `toronto_predictions_ensemble.csv`

### 3. `tabpfn_with_climate.py`
**Comparison script**
- Runs predictions **with and without** climate features
- Side-by-side comparison
- Shows improvement metrics

**Usage:**
```bash
python tabpfn_with_climate.py
```

**Output:**
- Baseline (no climate) results
- Enhanced (with climate) results
- Improvement analysis
- Saves: `toronto_predictions_climate.csv`

## Feature Detection Logic

All scripts use this logic:
```python
def prepare_features_target(df, use_climate=True):
    base_features = ['lat', 'long', 'alt', 'year']
    climate_features = ['spring_temp', 'spring_gdd',
                       'winter_chill_days', 'spring_precip']

    # Check if climate features exist
    has_climate = all(col in df.columns for col in climate_features)

    if use_climate and has_climate:
        # Use all 8 features
        features = base_features + climate_features
        # Drop records with missing climate data
        df_clean = df.dropna(subset=climate_features)
    else:
        # Use base 4 features only
        features = base_features
        df_clean = df

    return X, y, df_clean
```

## Expected Impact

### Without Climate Features (baseline):
- Toronto MAE: ~25 days
- Systematic underprediction

### With Climate Features (hypothesis):
- Should capture year-to-year variation
- Better handling of warm vs cold springs
- Expected MAE improvement: 30-50%

## Next Steps

Once enrichment completes:

1. **Test with climate features:**
   ```bash
   python tabpfn_with_climate.py
   ```

2. **Run full benchmark:**
   ```bash
   python tabpfn_cherry_blossom_prediction.py
   ```

3. **Compare results:**
   - Baseline (4 features): MAE ~25 days
   - Enhanced (8 features): MAE = ?

4. **Commit results:**
   ```bash
   git add data/*.csv toronto_predictions*.csv
   git commit -m "Add climate-enhanced predictions"
   git push
   ```

## Files Status

- ✅ `data/toronto.csv` - Enriched with climate data
- 🔄 `data/japan.csv` - Currently enriching (~10% done)
- ⏳ `data/kyoto.csv` - Pending
- ⏳ `data/liestal.csv` - Pending
- ⏳ `data/meteoswiss.csv` - Pending
- ⏳ `data/south_korea.csv` - Pending
- ⏳ `data/nyc.csv` - Pending
- ⏳ `data/vancouver.csv` - Pending
- ⏳ `data/washingtondc.csv` - Pending

## Background Process

Monitor enrichment progress:
```bash
# Check background job
ps aux | grep enrich

# View latest output
tail -f /path/to/logfile
```

Current process ID: f4ab74
