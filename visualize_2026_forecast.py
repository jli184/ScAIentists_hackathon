"""
Visualize 2026 Toronto Forecast
Clean, effective visual showing 3 climate scenarios with prediction intervals
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# Professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=1.1)

# Toronto brand colors
COLORS = {
    'toronto': '#003F87',
    'actual': '#E31837',
    'baseline': '#B8B8B8',
    'enhanced': '#00A651',
    'highlight': '#FDB913',
    'cold': '#4A90E2',
    'warm': '#FF6B35',
    'scenario_2025': '#8E44AD',  # Purple
    'scenario_2024': '#E67E22',  # Orange
    'scenario_2023': '#16A085'   # Teal
}

def load_forecast_data():
    """Load 2026 forecast data"""
    forecast_file = 'comparison_results/toronto_2026_forecast_full.csv'

    if not Path(forecast_file).exists():
        raise FileNotFoundError(
            f"Forecast data not found!\n"
            f"Run: python3 forecast_toronto_2026.py first"
        )

    forecast_df = pd.read_csv(forecast_file)
    print(f"✓ Loaded 2026 forecast with {len(forecast_df)} scenarios")

    return forecast_df

def create_2026_forecast_visual():
    """
    Create clean, effective forecast visualization
    Shows 3 scenarios with prediction intervals
    """
    forecast_df = load_forecast_data()

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('🌸 Toronto Cherry Blossom 2026 Forecast 🌸',
                 fontsize=28, fontweight='bold', y=0.98, color=COLORS['toronto'])

    # ========================================================================
    # PANEL 1: Main Forecast - Date Predictions with Uncertainty
    # ========================================================================
    ax = axes[0, 0]

    scenarios = forecast_df['climate_source_year'].values
    predictions = forecast_df['predicted_doy'].values
    q10 = forecast_df['predicted_doy_q10'].values
    q90 = forecast_df['predicted_doy_q90'].values

    scenario_colors = [COLORS['scenario_2025'], COLORS['scenario_2024'], COLORS['scenario_2023']]

    x_pos = np.arange(len(scenarios))

    # Plot predictions with error bars
    for i, (scenario, pred, low, high, color) in enumerate(zip(scenarios, predictions, q10, q90, scenario_colors)):
        ax.errorbar(i, pred, yerr=[[pred-low], [high-pred]],
                   fmt='o', markersize=20, linewidth=4, capsize=10, capthick=4,
                   color=color, label=f'{int(scenario)} Climate', zorder=5)

    # Format y-axis as dates
    y_ticks = ax.get_yticks()
    y_labels = []
    for doy in y_ticks:
        if 100 <= doy <= 150:
            date = datetime(2026, 1, 1) + timedelta(days=int(doy) - 1)
            y_labels.append(date.strftime('%b %d'))
        else:
            y_labels.append('')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Add mean line
    mean_pred = predictions.mean()
    ax.axhline(mean_pred, color=COLORS['toronto'], linestyle='--', linewidth=3,
              label=f'Mean: DOY {mean_pred:.1f}', alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{int(s)} Climate' for s in scenarios], fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Bloom Date', fontsize=16, fontweight='bold')
    ax.set_title('2026 Forecast by Climate Scenario', fontsize=20, fontweight='bold')
    ax.legend(fontsize=13, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.4, axis='y')

    # Add text box with summary
    mean_date = datetime(2026, 1, 1) + timedelta(days=int(mean_pred) - 1)
    text = f"Expected: {mean_date.strftime('%B %d')}\\n80% CI: ±{(q90.mean() - q10.mean())/2:.1f} days"
    ax.text(0.98, 0.02, text, transform=ax.transAxes,
           fontsize=15, fontweight='bold', verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor=COLORS['highlight'], alpha=0.8))

    # ========================================================================
    # PANEL 2: Climate Conditions Comparison
    # ========================================================================
    ax = axes[0, 1]

    # Normalize climate features for comparison
    climate_features = ['spring_temp', 'spring_gdd', 'winter_chill_days', 'spring_precip']
    feature_labels = ['Spring Temp\\n(°C)', 'Spring GDD', 'Winter Chill\\n(days)', 'Spring Precip\\n(mm)']

    x = np.arange(len(climate_features))
    width = 0.25

    for i, (scenario, color) in enumerate(zip(scenarios, scenario_colors)):
        row = forecast_df[forecast_df['climate_source_year'] == scenario].iloc[0]
        values = [row['spring_temp'], row['spring_gdd'], row['winter_chill_days'], row['spring_precip']]

        # Normalize for display
        normalized = []
        for val, feat in zip(values, climate_features):
            feat_col = forecast_df[feat]
            normalized.append((val - feat_col.min()) / (feat_col.max() - feat_col.min() + 0.001))

        offset = (i - 1) * width
        bars = ax.bar(x + offset, normalized, width, label=f'{int(scenario)}',
                     color=color, alpha=0.8, edgecolor='black', linewidth=2)

        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.1f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

    ax.set_ylabel('Normalized Value', fontsize=14, fontweight='bold')
    ax.set_title('Climate Conditions by Scenario', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, fontsize=12)
    ax.legend(title='Year', fontsize=12, title_fontsize=13)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # PANEL 3: Historical Context with 2026 Forecast
    # ========================================================================
    ax = axes[1, 0]

    # Load historical Toronto data
    comparison_file = 'comparison_results/toronto_comparison.csv'
    if Path(comparison_file).exists():
        historical = pd.read_csv(comparison_file)

        ax.plot(historical['year'], historical['bloom_doy'], 'o-',
               color=COLORS['actual'], linewidth=3, markersize=10,
               label='Historical Blooms', zorder=3)

        # Add 2026 forecast points
        for scenario, pred, color in zip(scenarios, predictions, scenario_colors):
            ax.scatter(2026, pred, s=400, marker='*', color=color,
                      edgecolor='black', linewidth=2, zorder=5,
                      label=f'2026 ({int(scenario)} climate)')

        # Add mean forecast
        ax.scatter(2026, mean_pred, s=500, marker='D', color=COLORS['toronto'],
                  edgecolor='black', linewidth=3, zorder=6, label='2026 Mean')

        ax.set_xlabel('Year', fontsize=16, fontweight='bold')
        ax.set_ylabel('Bloom Day of Year', fontsize=16, fontweight='bold')
        ax.set_title('2026 Forecast in Historical Context', fontsize=18, fontweight='bold')
        ax.legend(fontsize=11, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(historical['year'].min() - 1, 2027)
    else:
        ax.text(0.5, 0.5, 'Historical data not available\\nRun comparison first',
               ha='center', va='center', fontsize=16, transform=ax.transAxes)

    # ========================================================================
    # PANEL 4: Summary Statistics Table
    # ========================================================================
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary text
    summary = "2026 FORECAST SUMMARY\\n" + "="*40 + "\\n\\n"

    summary += "PREDICTIONS:\\n"
    for _, row in forecast_df.iterrows():
        scenario = int(row['climate_source_year'])
        pred_date = row['predicted_date']
        low_date = row['predicted_date_q10']
        high_date = row['predicted_date_q90']

        summary += f"  {scenario} Climate: {pred_date}\\n"
        summary += f"    (80% CI: {low_date} to {high_date})\\n\\n"

    mean_date = datetime(2026, 1, 1) + timedelta(days=int(mean_pred) - 1)
    summary += f"CONSENSUS FORECAST:\\n"
    summary += f"  Expected: {mean_date.strftime('%B %d, %Y')}\\n"
    summary += f"  Range: {forecast_df['predicted_date'].min()} to\\n"
    summary += f"         {forecast_df['predicted_date'].max()}\\n\\n"

    summary += "="*40 + "\\n"
    summary += "MODEL INFO:\\n"
    summary += f"  Training: {len(forecast_df)} climate scenarios\\n"
    summary += f"  Method: TabPFN with ALL data\\n"
    summary += f"  Features: Climate-enhanced (8)\\n"
    summary += f"  Uncertainty: 80% prediction interval\\n\\n"

    summary += "✓ Full Forecast (not cross-validation)\\n"
    summary += "✓ Includes Toronto historical data\\n"
    summary += "✓ Real climate scenarios (2023-2025)"

    ax.text(0.1, 0.95, summary,
           transform=ax.transAxes,
           fontsize=13, fontweight='bold',
           verticalalignment='top',
           family='monospace',
           bbox=dict(boxstyle='round,pad=1.5',
                    facecolor='white',
                    edgecolor=COLORS['toronto'],
                    linewidth=3))

    # Add footer
    fig.text(0.5, 0.01, '✓ Full Forecast Using ALL Available Data • Real Climate Scenarios • TabPFN Foundation Model',
            ha='center', fontsize=13, style='italic', color=COLORS['toronto'], fontweight='bold')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig('visuals/toronto_2026_forecast.png', dpi=300, bbox_inches='tight',
               facecolor='white')
    print("✓ Generated: visuals/toronto_2026_forecast.png")
    plt.close()

def main():
    print("="*70)
    print("Generating 2026 Toronto Forecast Visual")
    print("="*70)
    print()

    import os
    os.makedirs('visuals', exist_ok=True)

    try:
        create_2026_forecast_visual()

        print()
        print("="*70)
        print("✓ 2026 Forecast Visual Complete!")
        print("="*70)
        print()
        print("Generated: visuals/toronto_2026_forecast.png")
        print("• 2x2 panel layout")
        print("• Main forecast with uncertainty intervals")
        print("• Climate scenario comparison")
        print("• Historical context")
        print("• Summary statistics table")

    except FileNotFoundError as e:
        print(f"\\n✗ Error: {e}")
        print("\\nPlease run the forecast first:")
        print("  python3 forecast_toronto_2026.py")

if __name__ == '__main__':
    main()
