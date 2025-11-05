"""
Visualize cherry blossom predictions
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_toronto_predictions():
    """
    Create visualization of Toronto cherry blossom predictions
    """
    # Load prediction results
    try:
        results = pd.read_csv("toronto_predictions.csv")
    except FileNotFoundError:
        print("Error: toronto_predictions.csv not found. Run tabpfn_cherry_blossom_prediction.py first.")
        return

    # Set up the plot
    plt.figure(figsize=(14, 8))

    # Plot 1: Actual vs Predicted DOY over time
    plt.subplot(2, 2, 1)
    plt.plot(results['year'], results['bloom_doy'], 'o-', label='Actual', markersize=8, linewidth=2)
    plt.plot(results['year'], results['predicted_doy'], 's--', label='Predicted', markersize=8, linewidth=2, alpha=0.7)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Bloom Day of Year', fontsize=12)
    plt.title('Toronto Cherry Blossom: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot 2: Prediction Error over time
    plt.subplot(2, 2, 2)
    plt.bar(results['year'], results['error_days'], color='coral', alpha=0.7, edgecolor='black')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Prediction Error (days)', fontsize=12)
    plt.title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Plot 3: Scatter plot - Actual vs Predicted
    plt.subplot(2, 2, 3)
    plt.scatter(results['bloom_doy'], results['predicted_doy'], s=100, alpha=0.6, edgecolors='black')

    # Add diagonal line for perfect prediction
    min_val = min(results['bloom_doy'].min(), results['predicted_doy'].min())
    max_val = max(results['bloom_doy'].max(), results['predicted_doy'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)

    plt.xlabel('Actual Bloom DOY', fontsize=12)
    plt.ylabel('Predicted Bloom DOY', fontsize=12)
    plt.title('Actual vs Predicted (Scatter)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot 4: Error distribution
    plt.subplot(2, 2, 4)
    plt.hist(results['error_days'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.axvline(x=results['error_days'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean Error: {results["error_days"].mean():.1f}')
    plt.xlabel('Prediction Error (days)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('toronto_cherry_blossom_predictions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: toronto_cherry_blossom_predictions.png")

    # Print statistics
    print("\n" + "="*60)
    print("PREDICTION STATISTICS")
    print("="*60)
    print(f"Mean Absolute Error:     {abs(results['error_days']).mean():.2f} days")
    print(f"Mean Error (bias):       {results['error_days'].mean():.2f} days")
    print(f"Std Dev of Error:        {results['error_days'].std():.2f} days")
    print(f"Max Underestimation:     {results['error_days'].max():.2f} days")
    print(f"Max Overestimation:      {results['error_days'].min():.2f} days")

    # Focus on 2020-2025
    recent = results[results['year'] >= 2020]
    if len(recent) > 0:
        print(f"\n2020-2025 Predictions:")
        print(f"Mean Absolute Error:     {abs(recent['error_days']).mean():.2f} days")
        print(f"Mean Error (bias):       {recent['error_days'].mean():.2f} days")

if __name__ == "__main__":
    try:
        plot_toronto_predictions()
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")
