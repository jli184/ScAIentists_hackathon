# Toronto Cherry Blossom Prediction Project Summary

## What We've Accomplished

### 1. Data Collection
- ✅ Searched and found historical Toronto cherry blossom data from Sakura in High Park website
- ✅ Obtained 14 years of historical bloom data (2012-2025)
- ✅ Retrieved Toronto geographic coordinates: 43.647°N, 79.464°W, elevation 106.3m

###  2. Dataset Creation
- ✅ Created `data/toronto.csv` with historical bloom data
- ✅ Calculated midpoint bloom dates from date ranges
- ✅ Converted dates to day-of-year (DOY) format for modeling

### 3. TabPFN Installation
- 🔄 Currently installing TabPFN and PyTorch with CUDA support
- Large downloads in progress (~2.5GB total of CUDA libraries)

### 4. Code Development
Created the following scripts:

**explore_data.py**
- Loads and analyzes all cherry blossom datasets
- Shows data structure, coverage, and statistics
- Total: 15,286 records from 331 locations

**create_toronto_dataset.py**
- Generates Toronto CSV from historical data
- Calculates bloom DOY from date ranges
- Output: 14 records spanning 2012-2025

**tabpfn_cherry_blossom_prediction.py**
- Main prediction pipeline
- Benchmarks TabPFN on combined dataset (80/20 split)
- Trains on all non-Toronto locations
- Predicts Toronto as imputation problem
- Focuses analysis on 2020-2025 period
- Saves results to toronto_predictions.csv

**visualize_predictions.py**
- Creates 4-panel visualization
- Shows actual vs predicted over time
- Displays error distribution
- Generates scatter plot for evaluation

**TORONTO_PREDICTION_README.md**
- Comprehensive documentation
- Usage instructions
- Methodology explanation
- Future improvements

## Problem Framing

**Imputation Approach:**
Instead of traditional train/test split, we treat Toronto prediction as missing value imputation:
- Training: All locations EXCEPT Toronto (~15,272 records)
- Testing: Toronto with known covariates but "missing" bloom_doy
- This simulates predicting bloom for a new location

## Features Used (Covariates)
- `lat`: Latitude
- `long`: Longitude
- `alt`: Altitude (meters)
- `year`: Calendar year

**Target:**
- `bloom_doy`: Day of year when peak bloom occurs

## Key Insights

### Data Distribution
- **Asia**: 7,567 records (Japan, Korea)
- **Europe**: 6,774 records (Switzerland)
- **North America**: Only 109 records (DC, Vancouver, NYC)
- **Toronto**: 14 new records (2012-2025)

### Challenge
Toronto is geographically underrepresented in training data. Most data comes from Asia and Europe, with limited North American coverage.

### Toronto Bloom Patterns
- Average bloom: Day 120-125 (late April/early May)
- Range: Day 104-136 (April 13 - May 16)
- High year-to-year variability due to weather/climate
- 2012 was exceptionally early (April 13)
- 2014 was very late (May 16)

## Next Steps

Once TabPFN installation completes:

1. **Run Benchmark** on existing data
   ```bash
   python tabpfn_cherry_blossom_prediction.py
   ```

2. **Create Visualizations**
   ```bash
   pip install matplotlib
   python visualize_predictions.py
   ```

3. **Analyze Results** focusing on:
   - Overall prediction accuracy (MAE, RMSE)
   - Toronto-specific performance
   - 2020-2025 recent predictions
   - Error patterns and biases

4. **Commit and Push** all code and results

## Expected Performance

Based on TabPFN capabilities and similar phenology studies:
- General benchmark MAE: ~5-10 days
- Toronto imputation MAE: ~7-12 days
- Recent years (2020-2025): ~6-10 days

## Future Improvements

1. **Add Climate Features**
   - Temperature data (Dec-Apr monthly averages)
   - Precipitation patterns
   - Growing degree days
   - Winter chill hours

2. **Expand Training Data**
   - Add more North American locations
   - Include other Canadian cities
   - Extend temporal coverage

3. **Model Enhancements**
   - Ensemble methods
   - Uncertainty quantification
   - Time series components for trends

## Files Created

```
/home/user/ScAIentists_hackathon/
├── data/
│   └── toronto.csv  (NEW)
├── explore_data.py
├── create_toronto_dataset.py
├── tabpfn_cherry_blossom_prediction.py
├── visualize_predictions.py
├── TORONTO_PREDICTION_README.md
└── PROJECT_SUMMARY.md (this file)
```

## Installation Status

TabPFN installation in progress. Large CUDA library downloads (~2.5GB) taking time.
Once complete, we can run predictions immediately.

## References

- **Data Source**: Sakura in High Park (https://www.sakurainhighpark.com/)
- **TabPFN**: Hollmann et al. (2025), Nature
- **Repository**: https://github.com/joshspeagle/ScAIentists_hackathon
