"""
Visualization for Baseline vs Climate-Enhanced Comparison

Creates comprehensive visualizations showing:
1. Side-by-side prediction accuracy
2. Error distributions for both models
3. Improvement analysis
4. Year-by-year comparison
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

COLORS = {
    'baseline': '#DC143C',      # Crimson
    'climate': '#2E8B57',       # Sea green
    'actual': '#003F87',        # Dark blue
    'improvement': '#FFD700',   # Gold
    'perfect': '#696969'        # Dim gray
}

def load_comparison_results(location='toronto'):
    """Load comparison results for a location."""
    results_df = pd.read_csv(f'comparison_results/{location}_comparison.csv')
    with open(f'comparison_results/{location}_metrics.json', 'r') as f:
        metrics = json.load(f)
    return results_df, metrics

def create_comparison_visualization(location='toronto'):
    """Create comprehensive comparison visualization."""

    # Load data
    print(f"Loading comparison results for {location}...")
    results_df, metrics = load_comparison_results(location)

    print(f"  Baseline MAE: {metrics['metrics_baseline']['mae']:.2f} days")
    print(f"  Climate MAE:  {metrics['metrics_climate']['mae']:.2f} days")
    print(f"  Improvement:  {metrics['improvement']['mae']:+.2f} days")

    # Create figure with 6 panels
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{location.title()} Cherry Blossom Predictions: Baseline vs Climate-Enhanced',
                 fontsize=22, fontweight='bold', y=0.98)

    # Panel 1: Time series - Baseline
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(results_df['year'], results_df['bloom_doy'],
             'o-', color=COLORS['actual'], linewidth=2, markersize=8,
             label='Actual', alpha=0.9)
    ax1.plot(results_df['year'], results_df['predicted_baseline'],
             's--', color=COLORS['baseline'], linewidth=2, markersize=6,
             label='Baseline Pred', alpha=0.7)
    ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bloom DOY', fontsize=11, fontweight='bold')
    ax1.set_title('BASELINE: Geographic + Year Only', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'MAE: {metrics["metrics_baseline"]["mae"]:.2f} days',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 2: Time series - Climate-Enhanced
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(results_df['year'], results_df['bloom_doy'],
             'o-', color=COLORS['actual'], linewidth=2, markersize=8,
             label='Actual', alpha=0.9)
    ax2.plot(results_df['year'], results_df['predicted_climate'],
             'd--', color=COLORS['climate'], linewidth=2, markersize=6,
             label='Climate Pred', alpha=0.7)
    ax2.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Bloom DOY', fontsize=11, fontweight='bold')
    ax2.set_title('CLIMATE-ENHANCED: + Weather Data', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.05, 0.95, f'MAE: {metrics["metrics_climate"]["mae"]:.2f} days',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 3: Side-by-side comparison
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(results_df['year'], results_df['bloom_doy'],
             'o-', color=COLORS['actual'], linewidth=3, markersize=10,
             label='Actual', alpha=0.9, zorder=3)
    ax3.plot(results_df['year'], results_df['predicted_baseline'],
             's--', color=COLORS['baseline'], linewidth=2, markersize=6,
             label='Baseline', alpha=0.6, zorder=1)
    ax3.plot(results_df['year'], results_df['predicted_climate'],
             'd--', color=COLORS['climate'], linewidth=2, markersize=6,
             label='Climate', alpha=0.6, zorder=2)
    ax3.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Bloom DOY', fontsize=11, fontweight='bold')
    ax3.set_title('COMPARISON: All Three Overlaid', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Scatter - Baseline
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(results_df['bloom_doy'], results_df['predicted_baseline'],
                s=150, alpha=0.6, color=COLORS['baseline'], edgecolors='black', linewidth=1.5)
    min_val = min(results_df['bloom_doy'].min(), results_df['predicted_baseline'].min())
    max_val = max(results_df['bloom_doy'].max(), results_df['predicted_baseline'].max())
    ax4.plot([min_val, max_val], [min_val, max_val],
             '--', color=COLORS['perfect'], linewidth=2)
    ax4.set_xlabel('Actual Bloom DOY', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Predicted DOY', fontsize=11, fontweight='bold')
    ax4.set_title('Baseline: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.05, 0.95, f'R² = {metrics["metrics_baseline"]["r2"]:.3f}',
             transform=ax4.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 5: Scatter - Climate-Enhanced
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(results_df['bloom_doy'], results_df['predicted_climate'],
                s=150, alpha=0.6, color=COLORS['climate'], edgecolors='black', linewidth=1.5)
    min_val = min(results_df['bloom_doy'].min(), results_df['predicted_climate'].min())
    max_val = max(results_df['bloom_doy'].max(), results_df['predicted_climate'].max())
    ax5.plot([min_val, max_val], [min_val, max_val],
             '--', color=COLORS['perfect'], linewidth=2)
    ax5.set_xlabel('Actual Bloom DOY', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Predicted DOY', fontsize=11, fontweight='bold')
    ax5.set_title('Climate: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.text(0.05, 0.95, f'R² = {metrics["metrics_climate"]["r2"]:.3f}',
             transform=ax5.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 6: Improvement per year
    ax6 = plt.subplot(3, 3, 6)
    improvement = results_df['improvement'].values
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvement]
    ax6.bar(results_df['year'], improvement, color=colors, alpha=0.7, edgecolor='black')
    ax6.axhline(0, color='black', linestyle='-', linewidth=1)
    ax6.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Improvement (days)', fontsize=11, fontweight='bold')
    ax6.set_title('Year-by-Year Improvement', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    mean_improvement = improvement.mean()
    ax6.axhline(mean_improvement, color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {mean_improvement:+.2f} days')
    ax6.legend(fontsize=10)

    # Panel 7: Error distribution - Baseline
    ax7 = plt.subplot(3, 3, 7)
    errors_baseline = results_df['error_baseline']
    ax7.hist(errors_baseline, bins=15, color=COLORS['baseline'], alpha=0.7, edgecolor='black')
    ax7.axvline(0, color='black', linestyle='--', linewidth=2)
    ax7.axvline(errors_baseline.mean(), color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {errors_baseline.mean():.2f} days')
    ax7.set_xlabel('Error (days)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax7.set_title('Baseline Error Distribution', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')

    # Panel 8: Error distribution - Climate-Enhanced
    ax8 = plt.subplot(3, 3, 8)
    errors_climate = results_df['error_climate']
    ax8.hist(errors_climate, bins=15, color=COLORS['climate'], alpha=0.7, edgecolor='black')
    ax8.axvline(0, color='black', linestyle='--', linewidth=2)
    ax8.axvline(errors_climate.mean(), color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {errors_climate.mean():.2f} days')
    ax8.set_xlabel('Error (days)', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax8.set_title('Climate Error Distribution', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y')

    # Panel 9: Metrics comparison bar chart (including bias)
    ax9 = plt.subplot(3, 3, 9)
    metrics_names = ['MAE', 'RMSE', '|Bias|']
    baseline_vals = [
        metrics['metrics_baseline']['mae'],
        metrics['metrics_baseline']['rmse'],
        abs(metrics['metrics_baseline'].get('bias_mean', 0))
    ]
    climate_vals = [
        metrics['metrics_climate']['mae'],
        metrics['metrics_climate']['rmse'],
        abs(metrics['metrics_climate'].get('bias_mean', 0))
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax9.bar(x - width/2, baseline_vals, width, label='Baseline',
                    color=COLORS['baseline'], alpha=0.8, edgecolor='black')
    bars2 = ax9.bar(x + width/2, climate_vals, width, label='Climate',
                    color=COLORS['climate'], alpha=0.8, edgecolor='black')

    ax9.set_ylabel('Error (days)', fontsize=11, fontweight='bold')
    ax9.set_title('Metrics Comparison (Lower = Better)', fontsize=12, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics_names, fontsize=11, fontweight='bold')
    ax9.legend(fontsize=10, loc='upper left')
    ax9.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    os.makedirs('comparison_results', exist_ok=True)
    output_path = f'comparison_results/{location}_comparison_visual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Comparison visualization saved: {output_path}")

    plt.close()

def create_metrics_summary(location='toronto'):
    """Create a summary slide with key metrics."""

    results_df, metrics = load_comparison_results(location)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')

    # Title
    fig.text(0.5, 0.95, f'{location.title()} Prediction Comparison',
             ha='center', fontsize=26, fontweight='bold', color=COLORS['actual'])

    # Create comparison text
    has_climate = metrics.get('has_climate_features', True)
    baseline_features = ', '.join(metrics['features_baseline'])
    climate_features = ', '.join(metrics['features_climate'])

    improvement_mae = metrics['improvement']['mae']
    improvement_pct = (improvement_mae / metrics['metrics_baseline']['mae']) * 100

    if has_climate:
        climate_status = "✓ Climate data available and used"
        improvement_text = f"{improvement_mae:+.2f} days ({improvement_pct:+.1f}%)"
        if improvement_mae > 0:
            result_text = "✓ Climate data IMPROVES predictions"
        elif improvement_mae < 0:
            result_text = "⚠ Climate data DECREASES performance"
        else:
            result_text = "→ No change in performance"
    else:
        climate_status = "⚠ Climate data not yet available"
        improvement_text = "N/A (awaiting enrichment)"
        result_text = "Comparison pending climate data enrichment"

    summary_text = f"""
COMPARISON SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: TabPFN (Tabular Prior-Fitted Networks)
Target: {location.title()}
Status: {climate_status}

BASELINE MODEL (Geographic + Year)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features: {baseline_features}
Training samples: {metrics['metrics_baseline']['n_train']}
Test samples: {metrics['metrics_baseline']['n_test']}

Performance:
  • MAE:  {metrics['metrics_baseline']['mae']:.2f} days
  • RMSE: {metrics['metrics_baseline']['rmse']:.2f} days
  • R²:   {metrics['metrics_baseline']['r2']:.4f}
  • Bias (mean): {metrics['metrics_baseline'].get('bias_mean', 0):+.2f} days
  • Bias (median): {metrics['metrics_baseline'].get('bias_median', 0):+.2f} days

CLIMATE-ENHANCED MODEL (+ Weather Data)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features: {climate_features}
Training samples: {metrics['metrics_climate']['n_train']}
Test samples: {metrics['metrics_climate']['n_test']}

Performance:
  • MAE:  {metrics['metrics_climate']['mae']:.2f} days
  • RMSE: {metrics['metrics_climate']['rmse']:.2f} days
  • R²:   {metrics['metrics_climate']['r2']:.4f}
  • Bias (mean): {metrics['metrics_climate'].get('bias_mean', 0):+.2f} days
  • Bias (median): {metrics['metrics_climate'].get('bias_median', 0):+.2f} days

IMPROVEMENT (Climate vs Baseline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • MAE improvement:  {improvement_text}
  • RMSE improvement: {metrics['improvement']['rmse']:+.2f} days
  • R² improvement:   {metrics['improvement']['r2']:+.4f}

CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{result_text}

Both models use real TabPFN imputation - NO FAKE DATA.
Same train/test split ensures fair comparison.
"""

    fig.text(0.1, 0.80, summary_text,
             ha='left', va='top', fontsize=13, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Add timestamp
    from datetime import datetime
    timestamp = datetime.fromisoformat(metrics['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.5, 0.02, f'Generated: {timestamp}',
             ha='center', fontsize=10, style='italic', color='gray')

    # Save
    output_path = f'comparison_results/{location}_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Summary saved: {output_path}")

    plt.close()

def visualize_2026_forecasts(location='toronto'):
    """Create visualization for 2026 forecasts."""
    forecast_file = f'comparison_results/{location}_2026_forecasts.csv'

    if not Path(forecast_file).exists():
        print(f"No 2026 forecasts found for {location}")
        return

    forecast_df = pd.read_csv(forecast_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{location.title()} 2026 Cherry Blossom Forecast',
                 fontsize=20, fontweight='bold')

    # Panel 1: Bar chart of forecasts
    colors_map = {2025: '#2E8B57', 2024: '#4682B4', 2023: '#DAA520'}
    colors = [colors_map[y] for y in forecast_df['climate_source_year']]

    ax1.bar(forecast_df['climate_source_year'].astype(str) + ' Climate',
            forecast_df['predicted_doy'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Predicted Bloom Day of Year', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Climate Data Source', fontsize=13, fontweight='bold')
    ax1.set_title('2026 Forecasts Using Different Climate Years', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, row in forecast_df.iterrows():
        ax1.text(i, row['predicted_doy'] + 1, f"DOY {row['predicted_doy']:.0f}\n{row['predicted_date']}",
                ha='center', fontsize=11, fontweight='bold')

    # Add mean line
    mean_doy = forecast_df['predicted_doy'].mean()
    ax1.axhline(mean_doy, color='red', linestyle='--', linewidth=2, label=f'Mean: DOY {mean_doy:.0f}')
    ax1.legend(fontsize=11)

    # Panel 2: Climate conditions comparison
    ax2_data = forecast_df.set_index('climate_source_year')

    x = np.arange(len(forecast_df))
    width = 0.2

    # Normalize for visualization
    spring_temp_norm = ax2_data['spring_temp'] / ax2_data['spring_temp'].max() * 100
    spring_gdd_norm = ax2_data['spring_gdd'] / ax2_data['spring_gdd'].max() * 100
    chill_norm = ax2_data['winter_chill_days'] / ax2_data['winter_chill_days'].max() * 100
    precip_norm = ax2_data['spring_precip'] / ax2_data['spring_precip'].max() * 100

    ax2.bar(x - 1.5*width, spring_temp_norm, width, label='Spring Temp', alpha=0.8, edgecolor='black')
    ax2.bar(x - 0.5*width, spring_gdd_norm, width, label='Spring GDD', alpha=0.8, edgecolor='black')
    ax2.bar(x + 0.5*width, chill_norm, width, label='Winter Chill', alpha=0.8, edgecolor='black')
    ax2.bar(x + 1.5*width, precip_norm, width, label='Spring Precip', alpha=0.8, edgecolor='black')

    ax2.set_ylabel('Normalized Value (% of max)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Climate Source Year', fontsize=13, fontweight='bold')
    ax2.set_title('Climate Conditions Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ax2_data.index.astype(str), fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = f'comparison_results/{location}_2026_forecast_visual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 2026 forecast visualization saved: {output_path}")

    plt.close()

if __name__ == '__main__':
    location = 'toronto'

    print("="*70)
    print("Creating Comparison Visualizations")
    print("="*70)

    try:
        create_comparison_visualization(location)
        create_metrics_summary(location)

        # Check if 2026 forecasts exist and visualize them
        forecast_file = f'comparison_results/{location}_2026_forecasts.csv'
        if Path(forecast_file).exists():
            print("\nCreating 2026 forecast visualization...")
            visualize_2026_forecasts(location)

        print("\n" + "="*70)
        print("✓ Visualization complete!")
        print("="*70)
        print(f"\nGenerated files:")
        print(f"  - comparison_results/{location}_comparison_visual.png")
        print(f"  - comparison_results/{location}_summary.png")
        if Path(forecast_file).exists():
            print(f"  - comparison_results/{location}_2026_forecast_visual.png")

    except FileNotFoundError as e:
        print(f"\n✗ Error: Comparison results not found for {location}")
        print(f"  Run compare_baseline_vs_climate.py first to generate comparison data")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
