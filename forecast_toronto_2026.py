"""
Generate 2026 Toronto Forecast Using ALL Available Data

Unlike the comparison script (which leaves Toronto out for cross-validation),
this script trains on ALL data including Toronto's history to make the best
possible forecast for 2026.

Creates 3 forecast scenarios using climate data from 2025, 2024, and 2023.
"""
import pandas as pd
import numpy as np
from tabpfn import TabPFNRegressor
from datetime import datetime, timedelta
import json
import os
from data_utils import load_all_data

def forecast_2026_toronto():
    """
    Generate 2026 forecasts for Toronto using ALL available data.

    Returns:
        DataFrame with 3 forecast scenarios
    """
    print("="*70)
    print("TORONTO 2026 FORECAST")
    print("Training on ALL available data (including Toronto history)")
    print("="*70)

    # Load all data
    print("\n1. Loading all data...")
    all_data = load_all_data(include_city_files=True)

    # Prepare climate-enhanced dataset
    base_features = ['lat', 'long', 'alt', 'year']
    climate_features = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']
    all_features = base_features + climate_features

    # Filter to records with complete climate data
    train_data = all_data.dropna(subset=all_features + ['bloom_doy']).copy()

    print(f"  Total training records: {len(train_data)}")
    print(f"  Locations: {train_data['location'].nunique()}")
    print(f"  Year range: {train_data['year'].min()}-{train_data['year'].max()}")

    # Prepare training data
    X_train = train_data[all_features].values
    y_train = train_data['bloom_doy'].values

    # Sample if needed (TabPFN limit)
    max_train_samples = 10000
    if len(X_train) > max_train_samples:
        print(f"\n2. Sampling {max_train_samples} training records (TabPFN limit)...")
        indices = np.random.RandomState(42).choice(len(X_train), max_train_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
    else:
        print(f"\n2. Using all {len(X_train)} records for training...")

    # Train TabPFN model on ALL data
    print("\n3. Training TabPFN on ALL data (including Toronto)...")
    model = TabPFNRegressor(
        n_estimators=8,
        device='auto',
        random_state=42,
        ignore_pretraining_limits=True
    )
    model.fit(X_train, y_train)
    print("✓ Training complete")

    # Get Toronto location info
    toronto_mask = all_data['location'].str.lower().str.contains('toronto', na=False)
    toronto_data = all_data[toronto_mask].copy()

    if len(toronto_data) == 0:
        raise ValueError("No Toronto data found!")

    lat = toronto_data['lat'].iloc[0]
    lon = toronto_data['long'].iloc[0]
    alt = toronto_data['alt'].iloc[0]

    print(f"\n4. Generating 2026 forecasts for Toronto...")
    print(f"   Location: {lat:.4f}°N, {lon:.4f}°E, {alt:.1f}m")

    # Generate 3 forecast scenarios
    forecasts = []

    for ref_year in [2025, 2024, 2023]:
        ref_data = toronto_data[toronto_data['year'] == ref_year]

        if len(ref_data) == 0:
            print(f"   Warning: No data for {ref_year}, skipping forecast")
            continue

        # Check if climate data exists for this year
        if pd.isna(ref_data['spring_temp'].iloc[0]):
            print(f"   Warning: No climate data for {ref_year}, skipping forecast")
            continue

        # Create 2026 forecast using ref_year climate
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

        # Make prediction with uncertainty
        X_forecast = np.array([[
            lat, lon, alt, 2026,
            forecast_row['spring_temp'],
            forecast_row['spring_gdd'],
            forecast_row['winter_chill_days'],
            forecast_row['spring_precip']
        ]])

        # Mean prediction
        pred_doy = model.predict(X_forecast)[0]
        forecast_row['predicted_doy'] = pred_doy

        # Uncertainty quantiles
        quantile_preds = model.predict(X_forecast, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
        forecast_row['predicted_doy_q10'] = quantile_preds[0][0]
        forecast_row['predicted_doy_q50'] = quantile_preds[1][0]
        forecast_row['predicted_doy_q90'] = quantile_preds[2][0]

        # Convert DOY to date
        forecast_date = datetime(2026, 1, 1) + timedelta(days=int(pred_doy) - 1)
        forecast_row['predicted_date'] = forecast_date.strftime('%Y-%m-%d')

        # Uncertainty dates
        date_q10 = datetime(2026, 1, 1) + timedelta(days=int(forecast_row['predicted_doy_q10']) - 1)
        date_q90 = datetime(2026, 1, 1) + timedelta(days=int(forecast_row['predicted_doy_q90']) - 1)
        forecast_row['predicted_date_q10'] = date_q10.strftime('%Y-%m-%d')
        forecast_row['predicted_date_q90'] = date_q90.strftime('%Y-%m-%d')

        forecasts.append(forecast_row)

        print(f"\n   Scenario {ref_year} Climate:")
        print(f"      Spring temp: {forecast_row['spring_temp']:.2f}°C")
        print(f"      Spring GDD: {forecast_row['spring_gdd']:.1f}")
        print(f"      Winter chill: {forecast_row['winter_chill_days']:.0f} days")
        print(f"      Spring precip: {forecast_row['spring_precip']:.1f}mm")
        print(f"      → Predicted: {forecast_row['predicted_date']} (DOY {pred_doy:.1f})")
        print(f"      → 80% CI: {forecast_row['predicted_date_q10']} to {forecast_row['predicted_date_q90']}")

    if not forecasts:
        raise ValueError("Could not generate any forecasts (missing climate data)")

    forecast_df = pd.DataFrame(forecasts)

    # Save forecasts
    os.makedirs('comparison_results', exist_ok=True)
    output_csv = 'comparison_results/toronto_2026_forecast_full.csv'
    forecast_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved forecasts to: {output_csv}")

    # Summary statistics
    print(f"\n{'='*70}")
    print("2026 FORECAST SUMMARY")
    print(f"{'='*70}")

    mean_pred = forecast_df['predicted_doy'].mean()
    std_pred = forecast_df['predicted_doy'].std()
    min_pred = forecast_df['predicted_doy'].min()
    max_pred = forecast_df['predicted_doy'].max()

    mean_date = datetime(2026, 1, 1) + timedelta(days=int(mean_pred) - 1)
    min_date = datetime(2026, 1, 1) + timedelta(days=int(min_pred) - 1)
    max_date = datetime(2026, 1, 1) + timedelta(days=int(max_pred) - 1)

    print(f"\nBased on {len(forecasts)} climate scenarios:")
    print(f"  Mean prediction: {mean_date.strftime('%B %d, %Y')} (DOY {mean_pred:.1f} ±{std_pred:.1f})")
    print(f"  Range: {min_date.strftime('%b %d')} to {max_date.strftime('%b %d')} ({max_pred - min_pred:.1f} days)")

    # Show individual forecasts
    print(f"\nIndividual Forecasts:")
    for _, row in forecast_df.iterrows():
        print(f"  {int(row['climate_source_year'])} climate → {row['predicted_date']}")
        print(f"     80% CI: {row['predicted_date_q10']} to {row['predicted_date_q90']}")

    # Training data statistics
    print(f"\n{'='*70}")
    print("MODEL TRAINING INFO")
    print(f"{'='*70}")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {len(all_features)} ({', '.join(all_features)})")
    print(f"Model: TabPFN with {model.n_estimators} estimators")
    print(f"Includes Toronto historical data: YES")
    print(f"  (This is a FORECAST, not cross-validation)")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'location': 'toronto',
        'forecast_year': 2026,
        'n_scenarios': len(forecasts),
        'climate_source_years': [int(row['climate_source_year']) for _, row in forecast_df.iterrows()],
        'mean_prediction_doy': float(mean_pred),
        'std_prediction_doy': float(std_pred),
        'mean_prediction_date': mean_date.strftime('%Y-%m-%d'),
        'training_samples': len(X_train),
        'features': all_features,
        'includes_toronto_history': True,
        'forecasts': forecast_df.to_dict('records')
    }

    output_json = 'comparison_results/toronto_2026_forecast_full.json'
    with open(output_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata to: {output_json}")

    print(f"\n{'='*70}")
    print("✓ FORECAST COMPLETE")
    print(f"{'='*70}")

    return forecast_df, metadata

if __name__ == "__main__":
    forecast_df, metadata = forecast_2026_toronto()
