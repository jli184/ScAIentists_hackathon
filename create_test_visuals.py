#!/usr/bin/env python3
"""
Create test visualizations from saved prediction results.
Confirms the visualization pipeline works end-to-end.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'toronto': '#003F87',  # Toronto blue
    'actual': '#2E8B57',   # Sea green
    'predicted': '#DC143C', # Crimson
    'perfect': '#696969'    # Dim gray
}

def create_test_visuals():
    """Create test visualizations from saved predictions."""

    # Load saved test results
    print("Loading test results...")
    predictions_df = pd.read_csv('test_results/toronto_predictions_test.csv')
    with open('test_results/metrics_test.json', 'r') as f:
        metrics_data = json.load(f)

    print(f"Loaded {len(predictions_df)} predictions")
    print(f"Features used: {metrics_data['features']}")
    print(f"Training samples: {metrics_data['n_train']}")
    print(f"MAE: {metrics_data['metrics']['mae']:.2f} days")

    # Create figure with 4 panels
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('TabPFN Test Run: Toronto Cherry Blossom Predictions',
                 fontsize=20, fontweight='bold', y=0.98)

    # Panel 1: Time series of predictions
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(predictions_df['year'], predictions_df['bloom_doy'],
             'o-', color=COLORS['actual'], linewidth=2, markersize=8,
             label='Actual', alpha=0.8)
    ax1.plot(predictions_df['year'], predictions_df['predicted_doy'],
             's-', color=COLORS['predicted'], linewidth=2, markersize=8,
             label='Predicted', alpha=0.8)
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bloom Day of Year', fontsize=12, fontweight='bold')
    ax1.set_title('Predictions Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Actual vs Predicted scatter
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(predictions_df['bloom_doy'], predictions_df['predicted_doy'],
                s=150, alpha=0.6, color=COLORS['toronto'], edgecolors='black', linewidth=1.5)

    # Add perfect prediction line
    min_val = min(predictions_df['bloom_doy'].min(), predictions_df['predicted_doy'].min())
    max_val = max(predictions_df['bloom_doy'].max(), predictions_df['predicted_doy'].max())
    ax2.plot([min_val, max_val], [min_val, max_val],
             '--', color=COLORS['perfect'], linewidth=2, label='Perfect Prediction')

    ax2.set_xlabel('Actual Bloom DOY', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Bloom DOY', fontsize=12, fontweight='bold')
    ax2.set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add R² to plot
    from scipy.stats import pearsonr
    r, _ = pearsonr(predictions_df['bloom_doy'], predictions_df['predicted_doy'])
    r_squared = r**2
    ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}',
             transform=ax2.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 3: Error distribution
    ax3 = plt.subplot(2, 2, 3)
    errors = predictions_df['error_days']
    ax3.hist(errors, bins=10, color=COLORS['toronto'], alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.axvline(errors.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean Error: {errors.mean():.1f} days')
    ax3.set_xlabel('Prediction Error (days)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Year-by-year error bars
    ax4 = plt.subplot(2, 2, 4)
    years = predictions_df['year']
    actual = predictions_df['bloom_doy']
    predicted = predictions_df['predicted_doy']
    errors = predictions_df['error_days']

    # Create bar plot showing errors
    colors = ['green' if abs(e) < 10 else 'orange' if abs(e) < 20 else 'red' for e in errors]
    ax4.bar(years, abs(errors), color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Absolute Error (days)', fontsize=12, fontweight='bold')
    ax4.set_title('Year-by-Year Absolute Errors', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add MAE line
    ax4.axhline(metrics_data['metrics']['mae'], color='black', linestyle='--', linewidth=2,
                label=f'MAE: {metrics_data["metrics"]["mae"]:.1f} days')
    ax4.legend(fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = 'test_results/test_visuals.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Test visualization saved: {output_path}")

    # Create summary metrics visual
    create_metrics_summary(metrics_data, predictions_df)

    plt.close('all')

def create_metrics_summary(metrics, predictions_df):
    """Create a clean metrics summary slide."""

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Title
    fig.text(0.5, 0.95, 'Test Run Summary',
             ha='center', fontsize=24, fontweight='bold', color=COLORS['toronto'])

    # Create metrics text
    metrics_text = f"""
TEST CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: TabPFN (Tabular Prior-Fitted Network)
Features: {', '.join(metrics['features'])}
Training Samples: {metrics['n_train']}
Test Samples: {metrics['n_test']} (Toronto, 2012-2025)

PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mean Absolute Error (MAE): {metrics['metrics']['mae']:.2f} days
Root Mean Squared Error (RMSE): {metrics['metrics']['rmse']:.2f} days
R² Score: {metrics['metrics']['r2']:.3f}

NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• This is a baseline test with limited training data
• Toronto has climate features, other datasets being enriched
• Full performance expected after climate enrichment completes
• Current results use only: {', '.join(metrics['features'])}
"""

    fig.text(0.1, 0.75, metrics_text,
             ha='left', va='top', fontsize=14, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Add timestamp
    timestamp = datetime.fromisoformat(metrics['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')
    fig.text(0.5, 0.05, f'Generated: {timestamp}',
             ha='center', fontsize=10, style='italic', color='gray')

    # Save
    output_path = 'test_results/test_metrics_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Metrics summary saved: {output_path}")

    plt.close()

if __name__ == '__main__':
    print("Creating test visualizations from saved predictions...")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs('test_results', exist_ok=True)

    create_test_visuals()

    print("\n" + "=" * 60)
    print("✓ Test visualization pipeline complete!")
    print("\nGenerated files:")
    print("  - test_results/test_visuals.png (4-panel analysis)")
    print("  - test_results/test_metrics_summary.png (metrics slide)")
