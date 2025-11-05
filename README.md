# ScAIentists_hackathon

## 🌸 Predicting Toronto Cherry Blossom Peak Bloom

Our goal is to determine when the best time to take selfies with the cherry blossoms in Toronto's High Park using **TabPFN** (Tabular Prior-Fitted Network) with climate-enhanced features.

### 🎯 Approach

We use **TabPFN**, a foundation model for tabular data, to predict Toronto's cherry blossom peak bloom time by:

1. **Training on global data** - Use cherry blossom observations from Japan, USA, Switzerland, and South Korea (excluding Toronto)
2. **Climate feature enrichment** - Add year-specific weather features from Open-Meteo Historical Weather API:
   - Spring temperature (Jan-March average)
   - Spring Growing Degree Days (base 5°C)
   - Winter chill days (Dec-Feb days below 7°C)
   - Spring precipitation (Jan-March total)
3. **Leave-One-Location-Out (LOLO) imputation** - Predict Toronto using only location/year features (no bloom data)
4. **Ensemble predictions** - Generate robust predictions with TabPFN's built-in ensemble capability

### 📊 Data Sources

- **Cherry Blossom Data**: [GMU Cherry Blossom Competition](https://github.com/GMU-CherryBlossomCompetition/peak-bloom-prediction/tree/main/data)
- **Toronto Bloom Data**: Scraped from [Sakura in High Park](https://www.sakurainhighpark.com/)
- **Climate Data**: [Open-Meteo Historical Weather API](https://open-meteo.com/) (1940-present)

### 🔄 Complete Workflow

#### 1. Data Preparation
```bash
# Toronto data is pre-created with climate features
data/toronto.csv              # 14 years (2012-2025) with climate features
data/japan.csv                # 6,573 records (109 locations)
data/usa.csv                  # Historical US observations
data/south_korea.csv          # South Korean observations
data/switzerland.csv          # Swiss observations
```

#### 2. Climate Enrichment
```bash
# Enrich all CSVs with climate features
python enrich_csvs_optimized.py

# Features added:
# - spring_temp: Jan-March average temperature (°C)
# - spring_gdd: Growing Degree Days (base 5°C)
# - winter_chill_days: Dec-Feb days below 7°C
# - spring_precip: Jan-March total precipitation (mm)

# Progress is saved after each location to prevent data loss
```

#### 3. Model Training & Prediction
```bash
# Test run with small dataset (baseline)
python test_run_predictions.py
# Output: test_results/toronto_predictions_test.csv
#         test_results/metrics_test.json

# Full prediction with all data and climate features
python tabpfn_cherry_blossom_prediction.py
# Uses TabPFNRegressor with leave-Toronto-out approach
# Auto-detects climate features and uses enhanced model if available
```

#### 4. Visualization Generation
```bash
# Test visualizations from saved predictions
python create_test_visuals.py
# Output: test_results/test_visuals.png (4-panel analysis)
#         test_results/test_metrics_summary.png

# Toronto-focused presentation visuals
python create_toronto_focused_visuals.py
# Output: visuals/toronto_hero.png (multi-panel showcase)
#         visuals/toronto_summary_slide.png (clean slide)

# General analysis visuals
python create_presentation_visuals.py
# Output: visuals/main_comparison.png
#         visuals/climate_impact.png
```

### 📁 Key Files

**Data Files:**
- `data/toronto.csv` - Toronto bloom data with climate features (2012-2025)
- `data/japan.csv` - Japanese observations (being enriched)
- `data/usa.csv`, `data/south_korea.csv`, `data/switzerland.csv` - Additional training data

**Prediction Scripts:**
- `tabpfn_cherry_blossom_prediction.py` - Main prediction pipeline with auto-detection
- `test_run_predictions.py` - Test predictions with file saves
- `tabpfn_simple_prediction.py` - Simple ensemble approach (1000 samples)
- `tabpfn_with_climate.py` - Baseline vs climate comparison

**Enrichment Scripts:**
- `enrich_csvs_optimized.py` - Add climate features to all CSVs (saves after each location)

**Visualization Scripts:**
- `create_test_visuals.py` - Test visualization pipeline
- `create_toronto_focused_visuals.py` - Toronto-focused presentation visuals
- `create_presentation_visuals.py` - General analysis visuals

**Output Directories:**
- `test_results/` - Test predictions, metrics, and visuals
- `visuals/` - Final presentation-quality figures (300 DPI)

### 🚀 Quick Start

```bash
# 1. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only PyTorch
pip install tabpfn pandas numpy matplotlib seaborn scikit-learn

# 2. Test the pipeline (uses existing test data)
python test_run_predictions.py
python create_test_visuals.py

# 3. View results
ls -lh test_results/
# test_visuals.png - 4-panel analysis
# test_metrics_summary.png - metrics slide
# toronto_predictions_test.csv - predictions
# metrics_test.json - performance metrics
```

### 📈 Expected Results

**Baseline (4 features: lat, long, alt, year):**
- MAE: ~34 days
- Limited accuracy without climate context

**Climate-Enhanced (8 features: + spring_temp, spring_gdd, winter_chill_days, spring_precip):**
- Expected significant improvement in MAE
- Better capture of year-to-year variability
- More accurate Toronto predictions

### 🎓 References

- **TabPFN Paper**: [Hollmann et al., 2023](https://arxiv.org/abs/2207.01848)
- **Phenology Modeling**: [WUR-AI HybridML-Phenology](https://github.com/WUR-AI/HybridML-Phenology)
- **Cherry Blossom Competition**: [GMU Peak Bloom Prediction](https://github.com/GMU-CherryBlossomCompetition/peak-bloom-prediction) 
