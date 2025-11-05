"""
TabPFN Cherry Blossom Prediction with Climate Features
Enhanced with year-specific weather data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import glob

def load_all_data():
    """Load all cherry blossom datasets"""
    data_files = glob.glob("data/*.csv")
    all_dfs = [pd.read_csv(file) for file in data_files]
    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"Loaded {len(combined)} total records from {len(data_files)} files")
    print(f"Locations: {combined['location'].nunique()}")
    print(f"Year range: {combined['year'].min()}-{combined['year'].max()}")

    return combined

def prepare_features_target(df, use_climate=True):
    """
    Prepare features and target

    Features:
    - Basic: lat, long, alt, year
    - Climate (optional): spring_temp, spring_gdd, winter_chill_days, spring_precip
    """
    base_features = ['lat', 'long', 'alt', 'year']
    climate_features = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']

    if use_climate and all(col in df.columns for col in climate_features):
        features = base_features + climate_features
        # Only use records with complete climate data
        df_clean = df.dropna(subset=climate_features).copy()
        print(f"  Using {len(df_clean)}/{len(df)} records with complete climate data")
    else:
        features = base_features
        df_clean = df.copy()
        print(f"  Using {len(df_clean)} records (no climate filtering)")

    X = df_clean[features].values
    y = df_clean['bloom_doy'].values

    return X, y, df_clean

def predict_toronto(other_data, toronto_data, use_climate=True, n_samples=1000):
    """
    Predict Toronto using TabPFN

    Args:
        other_data: Non-Toronto locations
        toronto_data: Toronto data
        use_climate: Whether to use climate features
        n_samples: Training samples (for CPU efficiency)
    """
    from tabpfn import TabPFNRegressor

    # Prepare data
    X_train_all, y_train_all, train_df = prepare_features_target(other_data, use_climate)
    X_toronto, y_toronto_actual, toronto_df = prepare_features_target(toronto_data, use_climate)

    print(f"\nTraining set: {len(X_train_all)} samples")
    print(f"Toronto set: {len(X_toronto)} samples")

    # Sample if needed
    if len(X_train_all) > n_samples:
        np.random.seed(42)
        indices = np.random.choice(len(X_train_all), n_samples, replace=False)
        X_train = X_train_all[indices]
        y_train = y_train_all[indices]
        print(f"Sampled {n_samples} training samples")
    else:
        X_train = X_train_all
        y_train = y_train_all

    # Train TabPFN
    print("Training TabPFN...")
    model = TabPFNRegressor(
        n_estimators=8,
        device='auto',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict
    print(f"Predicting Toronto...")
    y_pred = model.predict(X_toronto)

    # Metrics
    mae = mean_absolute_error(y_toronto_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_toronto_actual, y_pred))
    r2 = r2_score(y_toronto_actual, y_pred)

    print(f"\nResults:")
    print(f"  MAE:  {mae:.2f} days")
    print(f"  RMSE: {rmse:.2f} days")
    print(f"  R²:   {r2:.4f}")

    return y_pred, y_toronto_actual, toronto_df, {'mae': mae, 'rmse': rmse, 'r2': r2}

def main():
    print("="*60)
    print("TabPFN Cherry Blossom Prediction with Climate Features")
    print("="*60)

    # Load data
    all_data = load_all_data()

    # Check for climate features
    climate_cols = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']
    has_climate = all(col in all_data.columns for col in climate_cols)

    if not has_climate:
        print("\n⚠️  Climate features not found in data!")
        print("Run enrich_csvs_optimized.py first to add climate data")
        return

    print(f"\n✓ Climate features detected")

    # Separate Toronto
    toronto_data = all_data[all_data['location'] == 'toronto'].copy()
    other_data = all_data[all_data['location'] != 'toronto'].copy()

    print(f"\nToronto records: {len(toronto_data)}")
    print(f"Other locations: {len(other_data)}")

    # ================================================================
    # COMPARISON 1: Without climate features (baseline)
    # ================================================================
    print("\n" + "="*60)
    print("BASELINE: Without Climate Features")
    print("="*60)

    y_pred_baseline, y_actual, toronto_df_baseline, metrics_baseline = predict_toronto(
        other_data, toronto_data, use_climate=False, n_samples=1000
    )

    # ================================================================
    # COMPARISON 2: With climate features
    # ================================================================
    print("\n" + "="*60)
    print("ENHANCED: With Climate Features")
    print("="*60)

    y_pred_enhanced, y_actual_enhanced, toronto_df_enhanced, metrics_enhanced = predict_toronto(
        other_data, toronto_data, use_climate=True, n_samples=1000
    )

    # ================================================================
    # RESULTS COMPARISON
    # ================================================================
    print("\n" + "="*60)
    print("COMPARISON: Baseline vs Enhanced")
    print("="*60)

    print(f"\nBaseline (no climate):")
    print(f"  MAE:  {metrics_baseline['mae']:.2f} days")
    print(f"  RMSE: {metrics_baseline['rmse']:.2f} days")
    print(f"  R²:   {metrics_baseline['r2']:.4f}")

    print(f"\nEnhanced (with climate):")
    print(f"  MAE:  {metrics_enhanced['mae']:.2f} days")
    print(f"  RMSE: {metrics_enhanced['rmse']:.2f} days")
    print(f"  R²:   {metrics_enhanced['r2']:.4f}")

    improvement_mae = metrics_baseline['mae'] - metrics_enhanced['mae']
    improvement_pct = (improvement_mae / metrics_baseline['mae']) * 100

    print(f"\nImprovement:")
    print(f"  MAE change: {improvement_mae:+.2f} days ({improvement_pct:+.1f}%)")
    print(f"  {'✓ Climate features help!' if improvement_mae > 0 else '✗ Climate features did not help'}")

    # Show detailed predictions
    print(f"\n{'='*60}")
    print("DETAILED PREDICTIONS (Enhanced Model)")
    print(f"{'='*60}")

    print(f"\n{'Year':<6} {'Actual':<8} {'Predicted':<10} {'Error':<8} {'Spring °C':<12}")
    print("-" * 50)
    for idx, row in toronto_df_enhanced.iterrows():
        year = int(row['year'])
        actual = int(row['bloom_doy'])
        pred_idx = toronto_df_enhanced.index.get_loc(idx)
        predicted = y_pred_enhanced[pred_idx]
        error = actual - predicted
        spring_temp = row.get('spring_temp', np.nan)

        print(f"{year:<6} {actual:<8} {predicted:<10.1f} {error:<8.1f} {spring_temp:<12.1f}")

    # Save results
    results = toronto_df_enhanced.copy()
    results['predicted_baseline'] = y_pred_baseline if len(y_pred_baseline) == len(results) else np.nan
    results['predicted_enhanced'] = y_pred_enhanced
    results['error_baseline'] = results['bloom_doy'] - results['predicted_baseline']
    results['error_enhanced'] = results['bloom_doy'] - results['predicted_enhanced']

    results.to_csv("toronto_predictions_climate.csv", index=False)
    print(f"\n✓ Saved predictions to: toronto_predictions_climate.csv")

    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
