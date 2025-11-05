# Toronto Cherry Blossom Prediction using TabPFN

## Overview

This project uses TabPFN (Tabular Prior-Fitted Network), a foundation model for tabular data, to predict cherry blossom peak bloom times in Toronto using an imputation approach. The model is trained on historical bloom data from locations worldwide (Japan, Korea, Switzerland, Washington DC, Vancouver, NYC) and then predicts Toronto's bloom dates as if they were missing values.

## Dataset

### Available Locations
- **Asia**: Japan (6,573 records), Kyoto (836 records), South Korea (994 records)
- **Europe**: Switzerland/MeteoSwiss (6,642 records), Liestal (132 records)
- **North America**: Washington DC (104 records), Vancouver (4 records), NYC (1 record)
- **Toronto**: 14 records (2012-2025) - newly created from Sakura in High Park data

### Features (Covariates)
- `lat`: Latitude
- `long`: Longitude
- `alt`: Altitude in meters
- `year`: Calendar year

### Target Variable
- `bloom_doy`: Day of year when peak bloom occurs (integer 1-366)

## Approach

### 1. Data Collection
- Scraped historical Toronto cherry blossom data from Sakura in High Park website
- Toronto coordinates: 43.646548°N, 79.463690°W, elevation 106.277m
- Peak bloom dates from 2012-2025 (midpoint of bloom period used)

### 2. Imputation Framework
Instead of traditional train/test splitting, we treat Toronto prediction as an **imputation problem**:
- **Training set**: All locations EXCEPT Toronto (~15,272 records)
- **Test set**: Toronto years with known covariates but "missing" bloom_doy

This simulates the realistic scenario where we have geographic/temporal data for a new location but no historical bloom records.

### 3. TabPFN Regressor
TabPFN is a transformer-based foundation model pre-trained on synthetic tabular datasets. Key advantages:
- No hyperparameter tuning required
- Fast inference (no iterative training)
- Strong performance on small-to-medium datasets
- Handles mixed feature types well

Configuration used:
```python
TabPFNRegressor(
    n_estimators=8,
    device='auto',
    random_state=42
)
```

## Files

### Data Files
- `data/toronto.csv` - Toronto historical bloom data (2012-2025)
- `data/*.csv` - Other location data (pre-existing)

### Scripts
- `create_toronto_dataset.py` - Generate Toronto CSV from scraped data
- `tabpfn_cherry_blossom_prediction.py` - Main prediction pipeline
  - Benchmark TabPFN on combined dataset (80/20 split)
  - Train on all non-Toronto data
  - Predict Toronto bloom dates
  - Focus analysis on 2020-2025 period
- `visualize_predictions.py` - Create plots of predictions vs actuals
- `explore_data.py` - Data exploration and statistics

## Usage

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn tabpfn matplotlib
```

### 2. Create Toronto Dataset
```bash
python create_toronto_dataset.py
```

### 3. Run Prediction Pipeline
```bash
python tabpfn_cherry_blossom_prediction.py
```

This will:
- Load all datasets (15,286 records total)
- Benchmark on general dataset (80/20 split)
- Train on non-Toronto locations
- Predict Toronto bloom dates for all years (2012-2025)
- Focus analysis on 2020-2025
- Save results to `toronto_predictions.csv`

### 4. Visualize Results
```bash
python visualize_predictions.py
```

Creates `toronto_cherry_blossom_predictions.png` with:
- Actual vs Predicted DOY over time
- Prediction error by year
- Scatter plot (actual vs predicted)
- Error distribution histogram

## Expected Results

Based on similar phenology prediction studies and TabPFN's capabilities:

### Performance Metrics
- **General Benchmark** (other locations, 80/20 split): MAE ~5-10 days
- **Toronto Imputation** (all years): MAE ~7-12 days
- **Toronto 2020-2025**: MAE ~6-10 days

### Key Insights
1. Geographic proximity matters - Toronto predictions benefit from Washington DC and Vancouver data
2. Year-to-year variability is high (climate change effects)
3. Simple features (lat/long/alt/year) capture major bloom timing patterns
4. TabPFN handles the irregular geographic distribution well

## Limitations

1. **Feature Engineering**: Only basic geographic + temporal features used
   - Missing: temperature, precipitation, winter chill hours, growing degree days
   - Could significantly improve predictions with climate data

2. **Sample Size**: Toronto has only 14 historical records
   - More years would allow better validation
   - 2016 had exceptional poor bloom (only 25% flowering)

3. **Temporal Drift**: Climate change is shifting bloom dates earlier
   - Model may not fully capture accelerating trends
   - Recent years may be less predictable

4. **Spatial Extrapolation**: Toronto is underrepresented in training data
   - Only 2 other nearby North American locations (DC, Vancouver)
   - Most training data from Asia and Europe

## Future Improvements

1. **Add Climate Covariates**
   - Monthly temperature averages (Dec-Apr)
   - Precipitation patterns
   - Growing degree days
   - Winter chill hours

2. **Feature Engineering**
   - Lag features (previous year's bloom date)
   - Climate indices (El Niño, NAO)
   - Urbanization metrics (urban heat island effect)

3. **Model Enhancements**
   - Ensemble TabPFN with other models (XGBoost, LightGBM)
   - Quantile regression for uncertainty estimates
   - Time series component for trend capture

4. **Data Augmentation**
   - Add more North American locations
   - Include other phenological events (leaf-out, first frost)
   - Extend historical record further back

## References

- **TabPFN Paper**: Hollmann et al. (2025), "Accurate predictions on small data with a tabular foundation model", Nature
- **Data Source**: Sakura in High Park (https://www.sakurainhighpark.com/)
- **TabPFN GitHub**: https://github.com/PriorLabs/TabPFN

## Citation

If using this code, please cite:
```
Toronto Cherry Blossom Prediction using TabPFN
ScAIentists Hackathon 2025
https://github.com/joshspeagle/ScAIentists_hackathon
```
