"""
Systematic Comparison: Baseline vs Climate-Enhanced TabPFN Predictions

This script runs TWO separate TabPFN predictions:
1. BASELINE: Using only geographic + year features (lat, long, alt, year)
2. ENHANCED: Adding climate features (spring_temp, spring_gdd, winter_chill_days, spring_precip)

Both use the SAME training/test split to ensure fair comparison.
All predictions use actual TabPFN imputation - NO FAKE DATA.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
import json
from datetime import datetime
import os
from data_utils import load_all_data

def prepare_dataset(df, use_climate=False):
    """
    Prepare features and target from dataframe.

    Args:
        df: Input dataframe
        use_climate: If True, use climate features; if False, use baseline only

    Returns:
        X, y, df_clean, feature_list
    """
    base_features = ['lat', 'long', 'alt', 'year']
    climate_features = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']

    if use_climate:
        # Check if climate features exist
        has_all_climate = all(col in df.columns for col in climate_features)

        if not has_all_climate:
            print(f"  WARNING: Climate features requested but not all available in data")
            print(f"  Available columns: {df.columns.tolist()}")
            print(f"  Falling back to baseline features only")
            feature_list = base_features
            df_clean = df.dropna(subset=base_features).copy()
        else:
            feature_list = base_features + climate_features
            # Only use records with complete climate data
            df_clean = df.dropna(subset=base_features + climate_features).copy()
            n_dropped = len(df) - len(df_clean)
            if n_dropped > 0:
                print(f"  Dropped {n_dropped} records with missing climate data")
    else:
        feature_list = base_features
        df_clean = df.dropna(subset=base_features).copy()

    X = df_clean[feature_list].values
    y = df_clean['bloom_doy'].values

    return X, y, df_clean, feature_list

def run_tabpfn_prediction(X_train, y_train, X_test, y_test, description, max_train_samples=10000):
    """
    Run TabPFN prediction and return metrics.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        description: Description of this run
        max_train_samples: Maximum training samples to use (TabPFN limit)

    Returns:
        predictions, metrics dict
    """
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

    # Sample training data if needed
    if len(X_train) > max_train_samples:
        print(f"Sampling {max_train_samples} training samples (TabPFN limit)...")
        indices = np.random.RandomState(42).choice(len(X_train), max_train_samples, replace=False)
        X_train_sampled = X_train[indices]
        y_train_sampled = y_train[indices]
    else:
        X_train_sampled = X_train
        y_train_sampled = y_train

    # Train TabPFN
    print("Training TabPFN...")
    model = TabPFNRegressor(
        n_estimators=8,
        device='auto',
        random_state=42,
        ignore_pretraining_limits=True
    )

    model.fit(X_train_sampled, y_train_sampled)
    print("✓ Training complete")

    # Predict with uncertainties
    print("Making predictions with uncertainty quantiles...")
    y_pred = model.predict(X_test)  # Mean prediction

    # Get quantile predictions for uncertainty
    quantile_preds = model.predict(X_test, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
    y_pred_q10 = quantile_preds[0]  # 10th percentile (lower bound)
    y_pred_q50 = quantile_preds[1]  # 50th percentile (median)
    y_pred_q90 = quantile_preds[2]  # 90th percentile (upper bound)

    print("✓ Predictions complete (with uncertainty)")

    # Calculate metrics
    errors = y_test - y_pred
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    bias_mean = np.mean(errors)
    bias_median = np.median(errors)

    print(f"\nResults:")
    print(f"  MAE:         {mae:.2f} days")
    print(f"  RMSE:        {rmse:.2f} days")
    print(f"  R²:          {r2:.4f}")
    print(f"  Bias (mean): {bias_mean:+.2f} days")
    print(f"  Bias (med):  {bias_median:+.2f} days")

    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'bias_mean': float(bias_mean),
        'bias_median': float(bias_median),
        'n_train': len(X_train_sampled),
        'n_test': len(X_test)
    }

    # Return predictions with uncertainties
    predictions = {
        'mean': y_pred,
        'q10': y_pred_q10,
        'q50': y_pred_q50,
        'q90': y_pred_q90
    }

    return predictions, metrics

def compare_baseline_vs_climate(target_location='toronto', max_train_samples=10000):
    """
    Run systematic comparison of baseline vs climate-enhanced predictions.

    Args:
        target_location: Location to predict (lowercase)
        max_train_samples: Maximum training samples

    Returns:
        results dict with both predictions
    """
    print("="*70)
    print("SYSTEMATIC COMPARISON: Baseline vs Climate-Enhanced TabPFN")
    print("="*70)
    print(f"\nTarget location: {target_location}")
    print(f"Max training samples: {max_train_samples}")

    # Load all data
    print("\n1. Loading data...")
    all_data = load_all_data(include_city_files=True)

    # Separate target location from training data
    target_mask = all_data['location'].str.lower().str.contains(target_location, na=False)
    target_data = all_data[target_mask].copy()
    train_data = all_data[~target_mask].copy()

    print(f"   Target ({target_location}): {len(target_data)} records")
    print(f"   Training (other locations): {len(train_data)} records")

    if len(target_data) == 0:
        raise ValueError(f"No data found for location: {target_location}")

    # ========================================================================
    # BASELINE PREDICTION (Geographic + Year only)
    # ========================================================================
    print(f"\n{'='*70}")
    print("BASELINE PREDICTION: Geographic + Year Features Only")
    print(f"{'='*70}")

    X_train_base, y_train_base, train_clean_base, features_base = prepare_dataset(
        train_data, use_climate=False
    )
    X_test_base, y_test_base, target_clean_base, _ = prepare_dataset(
        target_data, use_climate=False
    )

    print(f"Features: {', '.join(features_base)}")

    preds_baseline, metrics_baseline = run_tabpfn_prediction(
        X_train_base, y_train_base,
        X_test_base, y_test_base,
        "BASELINE MODEL (No Climate Data)"
    )

    # ========================================================================
    # CLIMATE-ENHANCED PREDICTION (Geographic + Year + Climate)
    # ========================================================================
    print(f"\n{'='*70}")
    print("CLIMATE-ENHANCED PREDICTION: Adding Climate Features")
    print(f"{'='*70}")

    X_train_climate, y_train_climate, train_clean_climate, features_climate = prepare_dataset(
        train_data, use_climate=True
    )
    X_test_climate, y_test_climate, target_clean_climate, _ = prepare_dataset(
        target_data, use_climate=True
    )

    print(f"Features: {', '.join(features_climate)}")

    # Check if we actually have climate data
    has_climate_features = len(features_climate) > len(features_base)

    if not has_climate_features:
        print("\n⚠️  WARNING: Climate features not available yet!")
        print("   Climate-enhanced prediction will be identical to baseline.")
        print("   Run enrichment scripts to add climate data first.")
        # Copy baseline predictions structure
        preds_climate = {k: v.copy() for k, v in preds_baseline.items()}
        metrics_climate = metrics_baseline.copy()
    else:
        preds_climate, metrics_climate = run_tabpfn_prediction(
            X_train_climate, y_train_climate,
            X_test_climate, y_test_climate,
            "CLIMATE-ENHANCED MODEL (With Climate Data)"
        )

    # ========================================================================
    # COMPARISON & RESULTS
    # ========================================================================
    print(f"\n{'='*70}")
    print("COMPARISON: Baseline vs Climate-Enhanced")
    print(f"{'='*70}")

    improvement_mae = metrics_baseline['mae'] - metrics_climate['mae']
    improvement_rmse = metrics_baseline['rmse'] - metrics_climate['rmse']
    improvement_r2 = metrics_climate['r2'] - metrics_baseline['r2']

    print(f"\nBASELINE Performance:")
    print(f"  MAE:         {metrics_baseline['mae']:.2f} days")
    print(f"  RMSE:        {metrics_baseline['rmse']:.2f} days")
    print(f"  R²:          {metrics_baseline['r2']:.4f}")
    print(f"  Bias (mean): {metrics_baseline['bias_mean']:+.2f} days")
    print(f"  Bias (med):  {metrics_baseline['bias_median']:+.2f} days")

    print(f"\nCLIMATE-ENHANCED Performance:")
    print(f"  MAE:         {metrics_climate['mae']:.2f} days")
    print(f"  RMSE:        {metrics_climate['rmse']:.2f} days")
    print(f"  R²:          {metrics_climate['r2']:.4f}")
    print(f"  Bias (mean): {metrics_climate['bias_mean']:+.2f} days")
    print(f"  Bias (med):  {metrics_climate['bias_median']:+.2f} days")

    print(f"\nIMPROVEMENT (Climate-Enhanced vs Baseline):")
    print(f"  MAE:  {improvement_mae:+.2f} days ({improvement_mae/metrics_baseline['mae']*100:+.1f}%)")
    print(f"  RMSE: {improvement_rmse:+.2f} days ({improvement_rmse/metrics_baseline['rmse']*100:+.1f}%)")
    print(f"  R²:   {improvement_r2:+.4f} ({improvement_r2/abs(metrics_baseline['r2'])*100:+.1f}%)")
    print(f"  Bias (mean): {metrics_baseline['bias_mean'] - metrics_climate['bias_mean']:+.2f} days")
    print(f"  Bias (med):  {metrics_baseline['bias_median'] - metrics_climate['bias_median']:+.2f} days")

    if has_climate_features:
        if improvement_mae > 0:
            print(f"\n✓ Climate data IMPROVES predictions by {improvement_mae:.2f} days MAE")
        elif improvement_mae < 0:
            print(f"\n⚠️  Climate data DECREASES performance by {abs(improvement_mae):.2f} days MAE")
        else:
            print(f"\n→ Climate data shows NO CHANGE in performance")
    else:
        print(f"\n⚠️  Climate comparison not yet possible - data needs enrichment")

    # Create results dataframe with predictions and uncertainties
    results_df = target_clean_base.copy()
    results_df['predicted_baseline'] = preds_baseline['mean']
    results_df['predicted_baseline_q10'] = preds_baseline['q10']
    results_df['predicted_baseline_q90'] = preds_baseline['q90']
    results_df['predicted_climate'] = preds_climate['mean']
    results_df['predicted_climate_q10'] = preds_climate['q10']
    results_df['predicted_climate_q90'] = preds_climate['q90']
    results_df['error_baseline'] = y_test_base - preds_baseline['mean']
    results_df['error_climate'] = y_test_climate - preds_climate['mean']
    results_df['abs_error_baseline'] = np.abs(results_df['error_baseline'])
    results_df['abs_error_climate'] = np.abs(results_df['error_climate'])
    results_df['improvement'] = results_df['abs_error_baseline'] - results_df['abs_error_climate']

    # Save results
    os.makedirs('comparison_results', exist_ok=True)

    output_csv = f'comparison_results/{target_location}_comparison.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved comparison to: {output_csv}")

    # Save metrics
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'target_location': target_location,
        'has_climate_features': has_climate_features,
        'features_baseline': features_base,
        'features_climate': features_climate,
        'metrics_baseline': metrics_baseline,
        'metrics_climate': metrics_climate,
        'improvement': {
            'mae': float(improvement_mae),
            'rmse': float(improvement_rmse),
            'r2': float(improvement_r2)
        }
    }

    output_json = f'comparison_results/{target_location}_metrics.json'
    with open(output_json, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"✓ Saved metrics to: {output_json}")

    print(f"\n{'='*70}")
    print("✓ COMPARISON COMPLETE")
    print(f"{'='*70}")

    return {
        'results_df': results_df,
        'metrics_baseline': metrics_baseline,
        'metrics_climate': metrics_climate,
        'comparison_data': comparison_data
    }

def generate_2026_forecasts(target_location='toronto', all_data=None, train_clean_climate=None,
                           model_climate=None, features_climate=None):
    """
    Generate 2026 forecasts using climate data from 2025, 2024, and 2023.

    Returns:
        DataFrame with 3 forecast scenarios
    """
    print(f"\n{'='*70}")
    print("2026 FORECASTS (Using Historical Climate Data)")
    print(f"{'='*70}")

    # Get target location data
    target_mask = all_data['location'].str.lower().str.contains(target_location, na=False)
    target_data = all_data[target_mask].copy()

    if len(target_data) == 0:
        print(f"No data found for {target_location}")
        return None

    # Get lat, lon, alt for the target location
    lat = target_data['lat'].iloc[0]
    lon = target_data['long'].iloc[0]
    alt = target_data['alt'].iloc[0]

    # Get climate data from 2025, 2024, 2023
    forecasts = []

    for ref_year in [2025, 2024, 2023]:
        ref_data = target_data[target_data['year'] == ref_year]

        if len(ref_data) == 0 or pd.isna(ref_data['spring_temp'].iloc[0]):
            print(f"  Warning: No climate data for {ref_year}, skipping this forecast")
            continue

        # Create 2026 prediction using ref_year climate
        forecast_row = {
            'year': 2026,
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'spring_temp': ref_data['spring_temp'].iloc[0],
            'spring_gdd': ref_data['spring_gdd'].iloc[0],
            'winter_chill_days': ref_data['winter_chill_days'].iloc[0],
            'spring_precip': ref_data['spring_precip'].iloc[0],
            'climate_source_year': ref_year
        }

        # Make prediction
        X_forecast = np.array([[
            lat, lon, alt, 2026,
            forecast_row['spring_temp'],
            forecast_row['spring_gdd'],
            forecast_row['winter_chill_days'],
            forecast_row['spring_precip']
        ]])

        pred_doy = model_climate.predict(X_forecast)[0]
        forecast_row['predicted_doy'] = pred_doy

        # Convert DOY to date
        from datetime import datetime, timedelta
        forecast_date = datetime(2026, 1, 1) + timedelta(days=int(pred_doy) - 1)
        forecast_row['predicted_date'] = forecast_date.strftime('%Y-%m-%d')

        forecasts.append(forecast_row)

        print(f"\n  Using {ref_year} climate data:")
        print(f"    Spring temp: {forecast_row['spring_temp']:.2f}°C")
        print(f"    Spring GDD: {forecast_row['spring_gdd']:.1f}")
        print(f"    Winter chill days: {forecast_row['winter_chill_days']:.0f}")
        print(f"    Spring precip: {forecast_row['spring_precip']:.1f}mm")
        print(f"    → Predicted DOY: {pred_doy:.1f} ({forecast_row['predicted_date']})")

    if not forecasts:
        print("\n  No forecasts could be generated (missing climate data)")
        return None

    forecast_df = pd.DataFrame(forecasts)

    # Save forecasts
    os.makedirs('comparison_results', exist_ok=True)
    output_csv = f'comparison_results/{target_location}_2026_forecasts.csv'
    forecast_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved 2026 forecasts to: {output_csv}")

    # Summary
    print(f"\n{'='*70}")
    print("2026 FORECAST SUMMARY")
    print(f"{'='*70}")
    mean_pred = forecast_df['predicted_doy'].mean()
    std_pred = forecast_df['predicted_doy'].std()
    print(f"Mean prediction: DOY {mean_pred:.1f} (±{std_pred:.1f} days)")
    print(f"Range: DOY {forecast_df['predicted_doy'].min():.1f} - {forecast_df['predicted_doy'].max():.1f}")
    print(f"Date range: {forecast_df['predicted_date'].min()} to {forecast_df['predicted_date'].max()}")

    return forecast_df

if __name__ == "__main__":
    # Run comparison for Toronto
    results = compare_baseline_vs_climate(
        target_location='toronto',
        max_train_samples=10000
    )

    # Generate 2026 forecasts if climate model available
    if 'comparison_data' in results and results['comparison_data']['has_climate_features']:
        print("\n" + "="*70)
        print("Generating 2026 forecasts...")
        print("="*70)

        # Need to reload data and retrain to get the model
        from data_utils import load_all_data
        all_data = load_all_data(include_city_files=True)
        target_location = 'toronto'

        target_mask = all_data['location'].str.lower().str.contains(target_location, na=False)
        train_data = all_data[~target_mask].copy()

        # Prepare climate training data
        base_features = ['lat', 'long', 'alt', 'year']
        climate_features = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']
        all_features = base_features + climate_features

        train_clean = train_data.dropna(subset=all_features).copy()
        X_train = train_clean[all_features].values
        y_train = train_clean['bloom_doy'].values

        # Sample if needed
        if len(X_train) > 10000:
            indices = np.random.RandomState(42).choice(len(X_train), 10000, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        # Train model
        print("Training model for 2026 forecasts...")
        model_climate = TabPFNRegressor(
            n_estimators=8,
            device='auto',
            random_state=42,
            ignore_pretraining_limits=True
        )
        model_climate.fit(X_train, y_train)

        # Generate forecasts
        forecast_df = generate_2026_forecasts(
            target_location=target_location,
            all_data=all_data,
            train_clean_climate=train_clean,
            model_climate=model_climate,
            features_climate=all_features
        )
    else:
        print("\n⚠️  Cannot generate 2026 forecasts: Climate features not available yet")

    # Auto-generate presentation visuals
    print("\n" + "="*70)
    print("Generating Presentation Visuals...")
    print("="*70)

    try:
        import subprocess
        result = subprocess.run(
            ['python3', 'create_hackathon_visuals.py'],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print(result.stdout)
            print("\n✓ Presentation visuals automatically generated!")
        else:
            print(f"Warning: Visualization generation had errors:\n{result.stderr}")
    except Exception as e:
        print(f"Warning: Could not auto-generate visuals: {e}")
        print("You can manually run: python3 create_hackathon_visuals.py")
