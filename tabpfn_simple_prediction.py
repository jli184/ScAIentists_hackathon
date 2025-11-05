"""
Simple TabPFN-based Cherry Blossom Prediction
Uses random sampling to stay within TabPFN's official limits (1000 samples on CPU)
"""
import pandas as pd
import numpy as np
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
    Extract features and target

    Features:
    - Base: lat, long, alt, year
    - Climate (if available): spring_temp, spring_gdd, winter_chill_days, spring_precip
    """
    base_features = ['lat', 'long', 'alt', 'year']
    climate_features = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']

    # Check if climate features are available
    has_climate = all(col in df.columns for col in climate_features)

    if use_climate and has_climate:
        features = base_features + climate_features
        # Only use records with complete climate data
        df_clean = df.dropna(subset=climate_features).copy()
        n_dropped = len(df) - len(df_clean)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} records with missing climate data")
    else:
        features = base_features
        df_clean = df.copy()
        if use_climate and not has_climate:
            print(f"  Climate features not available, using base features only")

    X = df_clean[features].values
    y = df_clean['bloom_doy'].values

    print(f"  Using {len(features)} features: {', '.join(features)}")

    return X, y, df_clean

def predict_toronto_simple(other_data, toronto_data, n_samples=1000, random_seed=42):
    """
    Predict Toronto using a random sample of other locations

    Args:
        other_data: DataFrame of non-Toronto locations
        toronto_data: DataFrame of Toronto data
        n_samples: Number of training samples (must be ≤1000 for CPU)
        random_seed: Random seed for reproducibility
    """
    from tabpfn import TabPFNRegressor

    # Prepare data
    X_train_all, y_train_all, other_data_clean = prepare_features_target(other_data)
    X_toronto, y_toronto_actual, toronto_data_clean = prepare_features_target(toronto_data)

    # Sample training data
    np.random.seed(random_seed)
    if len(X_train_all) > n_samples:
        indices = np.random.choice(len(X_train_all), n_samples, replace=False)
        X_train = X_train_all[indices]
        y_train = y_train_all[indices]
        print(f"Sampled {n_samples} from {len(X_train_all)} training samples")
    else:
        X_train = X_train_all
        y_train = y_train_all
        print(f"Using all {len(X_train)} training samples")

    # Train TabPFN
    print("Training TabPFN...")
    model = TabPFNRegressor(
        n_estimators=8,
        device='auto',
        random_state=random_seed
    )
    model.fit(X_train, y_train)

    # Predict Toronto
    print(f"Predicting Toronto bloom dates for {len(X_toronto)} years...")
    y_pred = model.predict(X_toronto)

    return y_pred, y_toronto_actual, toronto_data_clean

def predict_toronto_ensemble(other_data, toronto_data, n_models=10, n_samples=1000):
    """
    Predict Toronto using ensemble of models trained on different random samples

    Args:
        other_data: DataFrame of non-Toronto locations
        toronto_data: DataFrame of Toronto data
        n_models: Number of models in ensemble
        n_samples: Samples per model (≤1000 for CPU)
    """
    from tabpfn import TabPFNRegressor

    X_train_all, y_train_all, other_data_clean = prepare_features_target(other_data)
    X_toronto, y_toronto_actual, toronto_data_clean = prepare_features_target(toronto_data)

    print(f"\nTraining ensemble of {n_models} models...")
    print(f"Each model uses {n_samples} random samples")

    predictions = []

    for i in range(n_models):
        # Random sample for this model
        seed = 42 + i
        np.random.seed(seed)
        indices = np.random.choice(len(X_train_all), n_samples, replace=False)
        X_train = X_train_all[indices]
        y_train = y_train_all[indices]

        # Train model
        print(f"  Model {i+1}/{n_models}...", end=" ", flush=True)
        model = TabPFNRegressor(
            n_estimators=8,
            device='auto',
            random_state=seed
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_toronto)
        predictions.append(y_pred)
        print("done")

    # Average predictions
    y_pred_ensemble = np.mean(predictions, axis=0)
    y_std_ensemble = np.std(predictions, axis=0)

    return y_pred_ensemble, y_std_ensemble, y_toronto_actual, toronto_data_clean

def main():
    print("="*60)
    print("Simple TabPFN Cherry Blossom Prediction")
    print("="*60)

    # Load data
    all_data = load_all_data()

    # Separate Toronto
    toronto_data = all_data[all_data['location'] == 'toronto'].copy()
    other_data = all_data[all_data['location'] != 'toronto'].copy()

    print(f"\nToronto records: {len(toronto_data)}")
    print(f"Other locations: {len(other_data)}")

    # ================================================================
    # APPROACH 1: Single model with 1000 sample
    # ================================================================
    print("\n" + "="*60)
    print("APPROACH 1: Single Model (1000 samples)")
    print("="*60)

    y_pred_single, y_actual, toronto_data_clean = predict_toronto_simple(
        other_data, toronto_data, n_samples=1000, random_seed=42
    )

    mae_single = mean_absolute_error(y_actual, y_pred_single)
    rmse_single = np.sqrt(mean_squared_error(y_actual, y_pred_single))
    r2_single = r2_score(y_actual, y_pred_single)

    print(f"\nSingle Model Results:")
    print(f"  MAE:  {mae_single:.2f} days")
    print(f"  RMSE: {rmse_single:.2f} days")
    print(f"  R²:   {r2_single:.4f}")

    # Show predictions
    print(f"\n{'Year':<6} {'Actual':<8} {'Predicted':<10} {'Error':<10}")
    print("-" * 40)
    for i, row in enumerate(toronto_data_clean.itertuples()):
        error = y_actual[i] - y_pred_single[i]
        print(f"{int(row.year):<6} {int(y_actual[i]):<8} {y_pred_single[i]:<10.1f} {error:<10.1f}")

    # Save results
    results_single = toronto_data_clean.copy()
    results_single['predicted_doy'] = y_pred_single
    results_single['error_days'] = y_actual - y_pred_single
    results_single.to_csv("toronto_predictions_simple.csv", index=False)
    print(f"\n✓ Saved to: toronto_predictions_simple.csv")

    # ================================================================
    # APPROACH 2: Ensemble (optional - uncomment to run)
    # ================================================================
    run_ensemble = input("\nRun ensemble approach? (y/n): ").lower() == 'y'

    if run_ensemble:
        print("\n" + "="*60)
        print("APPROACH 2: Ensemble (10 models × 1000 samples)")
        print("="*60)

        y_pred_ensemble, y_std_ensemble, y_actual, toronto_data_clean_ens = predict_toronto_ensemble(
            other_data, toronto_data, n_models=10, n_samples=1000
        )

        mae_ensemble = mean_absolute_error(y_actual, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mean_squared_error(y_actual, y_pred_ensemble))
        r2_ensemble = r2_score(y_actual, y_pred_ensemble)

        print(f"\nEnsemble Results:")
        print(f"  MAE:  {mae_ensemble:.2f} days")
        print(f"  RMSE: {rmse_ensemble:.2f} days")
        print(f"  R²:   {r2_ensemble:.4f}")

        # Show predictions with uncertainty
        print(f"\n{'Year':<6} {'Actual':<8} {'Predicted':<10} {'Std Dev':<10} {'Error':<10}")
        print("-" * 50)
        for i, row in enumerate(toronto_data_clean_ens.itertuples()):
            error = y_actual[i] - y_pred_ensemble[i]
            print(f"{int(row.year):<6} {int(y_actual[i]):<8} {y_pred_ensemble[i]:<10.1f} "
                  f"{y_std_ensemble[i]:<10.2f} {error:<10.1f}")

        # Save ensemble results
        results_ensemble = toronto_data_clean_ens.copy()
        results_ensemble['predicted_doy'] = y_pred_ensemble
        results_ensemble['prediction_std'] = y_std_ensemble
        results_ensemble['error_days'] = y_actual - y_pred_ensemble
        results_ensemble.to_csv("toronto_predictions_ensemble.csv", index=False)
        print(f"\n✓ Saved to: toronto_predictions_ensemble.csv")

        # Comparison
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"Single Model MAE:   {mae_single:.2f} days")
        print(f"Ensemble MAE:       {mae_ensemble:.2f} days")
        print(f"Improvement:        {mae_single - mae_ensemble:+.2f} days")

    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
