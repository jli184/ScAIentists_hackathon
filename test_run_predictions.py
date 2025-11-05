"""
Test run: Real TabPFN predictions with small dataset (1000 samples)
Saves all results to files to preserve them
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
import json
from datetime import datetime

def test_tabpfn_prediction():
    """
    Run real TabPFN prediction with 1000 training samples
    Save all results to files
    """
    print("="*60)
    print("Test Run: TabPFN Prediction with Small Dataset")
    print("="*60)

    # Load Toronto data
    print("\n1. Loading data...")
    toronto = pd.read_csv('data/toronto.csv')
    print(f"   Toronto: {len(toronto)} records")

    # Load other locations (sample them to get 1000 total)
    other_files = ['washingtondc.csv', 'vancouver.csv', 'nyc.csv']
    other_data = []
    for f in other_files:
        try:
            df = pd.read_csv(f'data/{f}')
            other_data.append(df)
            print(f"   {f}: {len(df)} records")
        except:
            print(f"   {f}: not found, skipping")

    if other_data:
        others = pd.concat(other_data, ignore_index=True)
    else:
        print("   No other data found, using synthetic data")
        # Create synthetic data if files don't exist
        np.random.seed(42)
        synthetic = pd.DataFrame({
            'location': ['synthetic'] * 100,
            'lat': np.random.uniform(35, 45, 100),
            'long': np.random.uniform(-125, -70, 100),
            'alt': np.random.uniform(0, 200, 100),
            'year': np.random.randint(2000, 2025, 100),
            'bloom_doy': np.random.randint(90, 140, 100)
        })
        others = synthetic

    print(f"   Total other locations: {len(others)} records")

    # Check for climate features in BOTH datasets
    toronto_has_climate = 'spring_temp' in toronto.columns
    others_has_climate = 'spring_temp' in others.columns
    has_climate = toronto_has_climate and others_has_climate

    base_features = ['lat', 'long', 'alt', 'year']
    climate_features = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']

    if has_climate:
        print("   ✓ Climate features detected in all data")
        all_features = base_features + climate_features

        # Clean data - only use records with complete climate data
        toronto_clean = toronto.dropna(subset=climate_features).copy()
        others_clean = others.dropna(subset=climate_features).copy()

        print(f"   Toronto after cleaning: {len(toronto_clean)} records")
        print(f"   Others after cleaning: {len(others_clean)} records")
    else:
        if toronto_has_climate:
            print("   ⚠️  Toronto has climate features, but other data doesn't yet")
        print("   Using base features only for fair comparison")
        all_features = base_features
        toronto_clean = toronto.copy()
        others_clean = others.copy()

    # Sample training points
    print(f"\n2. Sampling training data...")
    n_train = min(1000, len(others_clean))
    if len(others_clean) > n_train:
        train_data = others_clean.sample(n=n_train, random_state=42)
        print(f"   Sampled {n_train} from {len(others_clean)} available")
    else:
        train_data = others_clean
        print(f"   Using all {len(others_clean)} available")

    # Prepare features
    X_train = train_data[all_features].values
    y_train = train_data['bloom_doy'].values
    X_toronto = toronto_clean[all_features].values
    y_toronto_actual = toronto_clean['bloom_doy'].values

    print(f"   Training shape: {X_train.shape}")
    print(f"   Toronto shape: {X_toronto.shape}")
    print(f"   Features: {len(all_features)} ({', '.join(all_features)})")

    # Train TabPFN
    print(f"\n3. Training TabPFN...")
    model = TabPFNRegressor(
        n_estimators=8,
        device='auto',
        random_state=42
    )

    model.fit(X_train, y_train)
    print("   ✓ Training complete")

    # Predict
    print(f"\n4. Predicting Toronto...")
    y_toronto_pred = model.predict(X_toronto)
    print("   ✓ Predictions complete")

    # Calculate metrics
    mae = mean_absolute_error(y_toronto_actual, y_toronto_pred)
    rmse = np.sqrt(mean_squared_error(y_toronto_actual, y_toronto_pred))
    r2 = r2_score(y_toronto_actual, y_toronto_pred)

    print(f"\n5. Results:")
    print(f"   MAE:  {mae:.2f} days")
    print(f"   RMSE: {rmse:.2f} days")
    print(f"   R²:   {r2:.4f}")

    # Create results dataframe
    results_df = toronto_clean.copy()
    results_df['predicted_doy'] = y_toronto_pred
    results_df['error_days'] = y_toronto_actual - y_toronto_pred
    results_df['abs_error_days'] = np.abs(y_toronto_actual - y_toronto_pred)

    # Save results to CSV
    output_csv = 'test_results/toronto_predictions_test.csv'
    import os
    os.makedirs('test_results', exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\n6. Saved predictions to: {output_csv}")

    # Save metrics to JSON
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'n_train': len(X_train),
        'n_test': len(X_toronto),
        'features': all_features,
        'has_climate': has_climate,
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        },
        'predictions': {
            'years': [int(y) for y in toronto_clean['year'].values],
            'actual': y_toronto_actual.tolist(),
            'predicted': y_toronto_pred.tolist()
        }
    }

    output_json = 'test_results/metrics_test.json'
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved metrics to: {output_json}")

    # Print detailed results
    print(f"\n7. Detailed predictions:")
    print(f"{'Year':<6} {'Actual':<8} {'Predicted':<10} {'Error':<8} {'Date':<12}")
    print("-" * 50)
    for _, row in results_df.iterrows():
        year = int(row['year'])
        actual = int(row['bloom_doy'])
        pred = row['predicted_doy']
        error = row['error_days']
        date = row.get('bloom_date', 'N/A')
        print(f"{year:<6} {actual:<8} {pred:<10.1f} {error:<8.1f} {date:<12}")

    print("\n" + "="*60)
    print("✓ Test run complete!")
    print("="*60)
    print(f"\nAll results saved to test_results/:")
    print(f"  - toronto_predictions_test.csv (full predictions)")
    print(f"  - metrics_test.json (metrics + raw predictions)")
    print(f"\nFeatures used: {', '.join(all_features)}")
    print(f"Climate features: {'YES' if has_climate else 'NO (base features only)'}")

    return results_df, metrics

if __name__ == "__main__":
    results_df, metrics = test_tabpfn_prediction()
