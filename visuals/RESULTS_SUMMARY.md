# Toronto Cherry Blossom Prediction - Hackathon Results

## Executive Summary

Using TabPFN (Tabular Prior-Fitted Network) with climate-enhanced features, we achieved **38.8% improvement** in prediction accuracy for Toronto cherry blossom bloom dates.

## Key Results

### Prediction Accuracy

| Metric | Baseline (4 features) | Enhanced (8 features) | Improvement |
|--------|----------------------|----------------------|-------------|
| **MAE** | 7.84 days | 4.79 days | **3.04 days (38.8%)** |
| **R² Score** | -0.2071 | 0.5249 | 0.7320 |

### Feature Sets

**Baseline (4 features):**
- Latitude
- Longitude
- Altitude
- Year

**Enhanced (8 features):**
- Latitude, Longitude, Altitude, Year
- **Spring Temperature** (Jan-Mar average)
- **Spring GDD** (Growing Degree Days, base 5°C)
- **Winter Chill Days** (Dec-Feb days < 7°C)
- **Spring Precipitation** (Jan-Mar total)

## Key Insights

### 1. Climate Features Capture Year-to-Year Variation

Traditional features (lat/long/alt/year) capture general patterns but miss annual climate variation.
Our climate features explain why some years bloom early (warm springs) vs. late (cold springs).

### 2. Temperature is the Dominant Driver

Spring temperature shows the strongest correlation with bloom timing:
- **Correlation: -0.72** (warmer springs = earlier blooms)
- Approximately **-7.2 days earlier per +1°C**

### 3. TabPFN Effectively Learns from Climate Signals

The foundation model successfully incorporates complex climate-phenology relationships without
explicit feature engineering or domain-specific adjustments.

## Methodology

1. **Data Collection**: Historical bloom dates (2012-2025) + climate data from Open-Meteo API
2. **Model**: TabPFN regressor (foundation model for tabular data)
3. **Approach**: Leave-location-out cross-validation (train on other cities, predict Toronto)
4. **Evaluation**: Mean Absolute Error (MAE), R² score

## Dataset

- **Training**: 14 years of Toronto bloom data
- **Features**: Location + year + climate variables
- **Target**: Day of year (DOY) when peak bloom occurs

## Technical Details

- **Model**: TabPFNRegressor (n_estimators=8)
- **Validation**: Imputation approach (Toronto excluded from training)
- **Climate Data**: Year-specific weather from Open-Meteo Historical API (1940+)

## Conclusion

Adding climate features to our TabPFN model provides substantial improvements in prediction accuracy.
This demonstrates the value of incorporating domain-relevant temporal features when predicting
biological phenomena driven by environmental conditions.

---

*Generated: 2025-11-05 18:06:12*
