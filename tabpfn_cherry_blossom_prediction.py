"""
TabPFN-based Cherry Blossom Peak Bloom Prediction
Using imputation approach to predict Toronto bloom dates
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import glob
import os

def load_all_data():
    """Load all cherry blossom datasets"""
    data_files = glob.glob("data/*.csv")

    all_dfs = []
    for file in data_files:
        df = pd.read_csv(file)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(combined)} total records from {len(data_files)} files")
    print(f"Locations: {combined['location'].nunique()}")
    print(f"Year range: {combined['year'].min()}-{combined['year'].max()}")

    return combined

def prepare_features_target(df):
    """
    Prepare features (X) and target (y) from dataframe
    Features: lat, long, alt, year
    Target: bloom_doy
    """
    X = df[['lat', 'long', 'alt', 'year']].values
    y = df['bloom_doy'].values
    return X, y

def benchmark_tabpfn(X_train, y_train, X_test, y_test, test_description="Test Set"):
    """
    Benchmark TabPFN regressor on provided train/test split
    """
    from tabpfn import TabPFNRegressor

    print(f"\n{'='*60}")
    print(f"Benchmarking TabPFN on {test_description}")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Initialize TabPFN regressor
    model = TabPFNRegressor(
        n_estimators=8,
        device='auto',
        random_state=42
    )

    # Train
    print("\nTraining TabPFN...")
    model.fit(X_train, y_train)

    # Predict
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{test_description} Results:")
    print(f"  MAE:  {mae:.2f} days")
    print(f"  RMSE: {rmse:.2f} days")
    print(f"  R²:   {r2:.4f}")

    # Show some example predictions
    print(f"\nSample predictions (first 10):")
    print(f"{'Actual':<10} {'Predicted':<12} {'Error':<10}")
    print("-" * 35)
    for i in range(min(10, len(y_test))):
        error = y_test[i] - y_pred[i]
        print(f"{y_test[i]:<10.0f} {y_pred[i]:<12.1f} {error:<10.1f}")

    return model, y_pred, {'mae': mae, 'rmse': rmse, 'r2': r2}

def main():
    print("="*60)
    print("TabPFN Cherry Blossom Peak Bloom Prediction")
    print("="*60)

    # Load all data
    all_data = load_all_data()

    # Separate Toronto data for imputation
    toronto_data = all_data[all_data['location'] == 'toronto'].copy()
    other_data = all_data[all_data['location'] != 'toronto'].copy()

    print(f"\nToronto records: {len(toronto_data)}")
    print(f"Other locations: {len(other_data)}")

    # ================================================================
    # BENCHMARK 1: General performance on combined dataset
    # ================================================================
    print("\n" + "="*60)
    print("BENCHMARK 1: General Performance (80/20 split)")
    print("="*60)

    X_all, y_all = prepare_features_target(other_data)
    X_train_bench, X_test_bench, y_train_bench, y_test_bench = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    _, _, metrics_bench = benchmark_tabpfn(
        X_train_bench, y_train_bench,
        X_test_bench, y_test_bench,
        "Combined Dataset (80/20 split)"
    )

    # ================================================================
    # TORONTO IMPUTATION: Predict Toronto using all other locations
    # ================================================================
    print("\n" + "="*60)
    print("TORONTO IMPUTATION: Predicting Toronto bloom dates")
    print("="*60)

    # Prepare training data (all non-Toronto locations)
    X_train_toronto, y_train_toronto = prepare_features_target(other_data)

    # Prepare Toronto data for prediction
    X_toronto, y_toronto_actual = prepare_features_target(toronto_data)

    # Train on all non-Toronto data
    from tabpfn import TabPFNRegressor

    print(f"\nTraining on {len(X_train_toronto)} samples from other locations...")
    model_toronto = TabPFNRegressor(
        n_estimators=8,
        device='auto',
        random_state=42
    )
    model_toronto.fit(X_train_toronto, y_train_toronto)

    # Predict Toronto
    print(f"Predicting Toronto bloom dates for {len(X_toronto)} years...")
    y_toronto_pred = model_toronto.predict(X_toronto)

    # Create results dataframe
    toronto_results = toronto_data.copy()
    toronto_results['predicted_doy'] = y_toronto_pred
    toronto_results['error_days'] = toronto_results['bloom_doy'] - y_toronto_pred

    # Calculate metrics
    mae_toronto = mean_absolute_error(y_toronto_actual, y_toronto_pred)
    rmse_toronto = np.sqrt(mean_squared_error(y_toronto_actual, y_toronto_pred))
    r2_toronto = r2_score(y_toronto_actual, y_toronto_pred)

    print(f"\n{'='*60}")
    print("TORONTO PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"MAE:  {mae_toronto:.2f} days")
    print(f"RMSE: {rmse_toronto:.2f} days")
    print(f"R²:   {r2_toronto:.4f}")

    print(f"\n{'Year':<6} {'Actual DOY':<12} {'Predicted DOY':<15} {'Error (days)':<15} {'Bloom Date':<15}")
    print("-" * 75)
    for _, row in toronto_results.iterrows():
        print(f"{int(row['year']):<6} {int(row['bloom_doy']):<12} "
              f"{row['predicted_doy']:<15.1f} {row['error_days']:<15.1f} "
              f"{row['bloom_date']:<15}")

    # Focus on 2020-2025 period
    toronto_2020_2025 = toronto_results[toronto_results['year'] >= 2020]

    print(f"\n{'='*60}")
    print("FOCUS: Toronto 2020-2025 Predictions")
    print(f"{'='*60}")

    if len(toronto_2020_2025) > 0:
        mae_recent = mean_absolute_error(
            toronto_2020_2025['bloom_doy'],
            toronto_2020_2025['predicted_doy']
        )
        rmse_recent = np.sqrt(mean_squared_error(
            toronto_2020_2025['bloom_doy'],
            toronto_2020_2025['predicted_doy']
        ))

        print(f"MAE (2020-2025):  {mae_recent:.2f} days")
        print(f"RMSE (2020-2025): {rmse_recent:.2f} days")

        print(f"\n{'Year':<6} {'Actual DOY':<12} {'Predicted DOY':<15} {'Error (days)':<15}")
        print("-" * 55)
        for _, row in toronto_2020_2025.iterrows():
            print(f"{int(row['year']):<6} {int(row['bloom_doy']):<12} "
                  f"{row['predicted_doy']:<15.1f} {row['error_days']:<15.1f}")

    # Save results
    output_file = "toronto_predictions.csv"
    toronto_results.to_csv(output_file, index=False)
    print(f"\n✓ Saved Toronto predictions to: {output_file}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"General Benchmark (other locations): MAE = {metrics_bench['mae']:.2f} days")
    print(f"Toronto Imputation (all years):      MAE = {mae_toronto:.2f} days")
    if len(toronto_2020_2025) > 0:
        print(f"Toronto 2020-2025:                    MAE = {mae_recent:.2f} days")

    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
