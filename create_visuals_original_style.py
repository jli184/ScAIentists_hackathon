"""
Generate Hackathon Visuals in ORIGINAL 3x3 Style
Uses REAL TabPFN predictions from comparison results with uncertainties
Matches the exact layout and appearance of create_toronto_focused_visuals.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=1.1)

# Toronto brand colors (EXACTLY as in original)
COLORS = {
    'toronto': '#003F87',        # Toronto blue
    'actual': '#E31837',         # Vibrant red
    'baseline': '#B8B8B8',       # Gray
    'enhanced': '#00A651',       # Green
    'highlight': '#FDB913',      # Gold
    'cold': '#4A90E2',          # Light blue
    'warm': '#FF6B35'           # Orange
}

def load_real_comparison_results():
    """Load real TabPFN comparison results (not mock data!)"""
    comparison_file = 'comparison_results/toronto_comparison.csv'
    metrics_file = 'comparison_results/toronto_metrics.json'

    if not Path(comparison_file).exists():
        raise FileNotFoundError(
            f"Comparison results not found!\n"
            f"Run: python3 compare_baseline_vs_climate.py first"
        )

    results_df = pd.read_csv(comparison_file)
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    print("✓ Loaded REAL TabPFN predictions (not mock data!)")
    print(f"  {len(results_df)} years of Toronto data")
    print(f"  Baseline MAE: {metrics['metrics_baseline']['mae']:.2f} days")
    print(f"  Climate MAE: {metrics['metrics_climate']['mae']:.2f} days")

    return results_df, metrics

def create_toronto_hero_original_style():
    """
    Hero figure matching ORIGINAL 3x3 layout with REAL predictions
    """
    results_df, metrics = load_real_comparison_results()

    fig = plt.figure(figsize=(24, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.95, top=0.93, bottom=0.07)

    years = results_df['year'].values
    actual_doy = results_df['bloom_doy'].values
    baseline_pred = results_df['predicted_baseline'].values
    enhanced_pred = results_df['predicted_climate'].values

    # Get uncertainty bounds if available
    has_uncertainty = 'predicted_climate_q10' in results_df.columns
    if has_uncertainty:
        enhanced_q10 = results_df['predicted_climate_q10'].values
        enhanced_q90 = results_df['predicted_climate_q90'].values

    mae_baseline = metrics['metrics_baseline']['mae']
    mae_enhanced = metrics['metrics_climate']['mae']

    # Convert DOY to dates for labels
    actual_dates = []
    for year, doy in zip(years, actual_doy):
        date = datetime(int(year), 1, 1) + timedelta(days=int(doy)-1)
        actual_dates.append(date.strftime('%b %d'))

    # ========================================================================
    # MAIN PANEL: Toronto Predictions (spans 2 rows, 2 cols) - TOP LEFT
    # ========================================================================
    ax_main = fig.add_subplot(gs[0:2, 0:2])

    # Plot uncertainty band if available
    if has_uncertainty:
        ax_main.fill_between(years, enhanced_q10, enhanced_q90,
                            color=COLORS['enhanced'], alpha=0.2, label='Climate 80% CI')

    # Plot predictions
    ax_main.plot(years, actual_doy, 'o-', color=COLORS['actual'],
                linewidth=4, markersize=14, label='Actual Toronto Bloom',
                zorder=5, markeredgewidth=2, markeredgecolor='white')

    ax_main.plot(years, baseline_pred, 's--', color=COLORS['baseline'],
                linewidth=3, markersize=10, alpha=0.6, label='Baseline (No Climate)',
                zorder=3)

    ax_main.plot(years, enhanced_pred, '^-', color=COLORS['enhanced'],
                linewidth=3, markersize=10, alpha=0.9, label='Enhanced (With Climate)',
                zorder=4)

    # Highlight coldest and warmest years (based on spring_temp)
    if 'spring_temp' in results_df.columns:
        coldest_idx = results_df['spring_temp'].idxmin()
        warmest_idx = results_df['spring_temp'].idxmax()

        # Cold year annotation
        ax_main.scatter(years[coldest_idx], actual_doy[coldest_idx],
                       s=500, facecolors='none', edgecolors=COLORS['cold'],
                       linewidth=4, zorder=6)
        ax_main.annotate(f'{int(years[coldest_idx])}: Cold Spring\\nLate Bloom ({actual_dates[coldest_idx]})',
                        xy=(years[coldest_idx], actual_doy[coldest_idx]),
                        xytext=(years[coldest_idx]-2, actual_doy[coldest_idx]+12),
                        fontsize=13, fontweight='bold', color=COLORS['cold'],
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                 edgecolor=COLORS['cold'], linewidth=2),
                        arrowprops=dict(arrowstyle='->', color=COLORS['cold'], lw=3))

        # Warm year annotation
        ax_main.scatter(years[warmest_idx], actual_doy[warmest_idx],
                       s=500, facecolors='none', edgecolors=COLORS['warm'],
                       linewidth=4, zorder=6)
        ax_main.annotate(f'{int(years[warmest_idx])}: Warm Spring\\nEarly Bloom ({actual_dates[warmest_idx]})',
                        xy=(years[warmest_idx], actual_doy[warmest_idx]),
                        xytext=(years[warmest_idx]+2, actual_doy[warmest_idx]-12),
                        fontsize=13, fontweight='bold', color=COLORS['warm'],
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                 edgecolor=COLORS['warm'], linewidth=2),
                        arrowprops=dict(arrowstyle='->', color=COLORS['warm'], lw=3))

    ax_main.set_xlabel('Year', fontsize=18, fontweight='bold')
    ax_main.set_ylabel('Peak Bloom Day of Year', fontsize=18, fontweight='bold')
    ax_main.set_title('🌸 Predicting Toronto Cherry Blossom Peak Bloom with TabPFN 🌸',
                     fontsize=22, fontweight='bold', pad=20, color=COLORS['toronto'])
    ax_main.legend(fontsize=15, loc='upper left', framealpha=0.95,
                  edgecolor=COLORS["toronto"])
    ax_main.grid(True, alpha=0.3, linewidth=1.5)
    ax_main.set_ylim(actual_doy.min()-10, actual_doy.max()+15)

    # ========================================================================
    # TOP RIGHT: Prediction Performance - JUST MAE
    # ========================================================================
    ax_perf = fig.add_subplot(gs[0, 2])

    improvement = mae_baseline - mae_enhanced
    improvement_pct = (improvement / mae_baseline) * 100

    metrics_names = ['MAE\\n(days)']
    baseline_vals = [mae_baseline]
    enhanced_vals = [mae_enhanced]

    x = np.arange(len(metrics_names))
    width = 0.6

    bars1 = ax_perf.bar(x - width/2, baseline_vals, width,
                       label='Baseline', color=COLORS['baseline'],
                       alpha=0.8, edgecolor='black', linewidth=3)
    bars2 = ax_perf.bar(x + width/2, enhanced_vals, width,
                       label='Enhanced', color=COLORS['enhanced'],
                       alpha=0.9, edgecolor='black', linewidth=3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax_perf.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=18, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax_perf.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=18, fontweight='bold',
                    color=COLORS['enhanced'])

    ax_perf.set_ylabel('Error (days)', fontsize=16, fontweight='bold')
    ax_perf.set_title(f'🎯 {improvement_pct:.0f}% Improvement',
                     fontsize=18, fontweight='bold', color=COLORS['enhanced'])
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(metrics_names, fontsize=14)
    ax_perf.legend(fontsize=12, loc='upper right')
    ax_perf.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # MIDDLE RIGHT: Year-by-Year Toronto Predictions (Recent 6 years)
    # ========================================================================
    ax_years = fig.add_subplot(gs[1, 2])

    # Show recent 6 years
    recent_years = years[-6:]
    recent_actual = actual_doy[-6:]
    recent_enhanced = enhanced_pred[-6:]
    recent_dates = actual_dates[-6:]

    y_pos = np.arange(len(recent_years))

    ax_years.barh(y_pos, recent_enhanced, height=0.6,
                 color=COLORS['enhanced'], alpha=0.7,
                 edgecolor='black', linewidth=2, label='Predicted')

    # Add actual as markers
    ax_years.scatter(recent_actual, y_pos, s=200, color=COLORS['actual'],
                    marker='D', zorder=5, edgecolor='white', linewidth=2,
                    label='Actual')

    # Add date labels
    for i, (pred, actual, date) in enumerate(zip(recent_enhanced, recent_actual, recent_dates)):
        error = actual - pred
        ax_years.text(max(pred, actual) + 3, i,
                     f'{date}\\n(err: {error:+.0f}d)',
                     va='center', fontsize=11, fontweight='bold')

    ax_years.set_yticks(y_pos)
    ax_years.set_yticklabels([f'{int(y)}' for y in recent_years], fontsize=13)
    ax_years.set_xlabel('Day of Year', fontsize=14, fontweight='bold')
    ax_years.set_title('Recent Toronto Predictions', fontsize=16, fontweight='bold')
    ax_years.legend(fontsize=11, loc='lower right')
    ax_years.grid(True, alpha=0.3, axis='x')
    ax_years.invert_yaxis()

    # ========================================================================
    # BOTTOM LEFT: Climate Impact on Toronto
    # ========================================================================
    ax_climate = fig.add_subplot(gs[2, 0])

    if 'spring_temp' in results_df.columns:
        scatter = ax_climate.scatter(results_df['spring_temp'], actual_doy,
                                    s=300, c=years, cmap='viridis',
                                    alpha=0.8, edgecolors='black', linewidth=2)

        # Trend line
        z = np.polyfit(results_df['spring_temp'], actual_doy, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(results_df['spring_temp'].min(),
                             results_df['spring_temp'].max(), 100)
        ax_climate.plot(x_trend, p(x_trend), "r--", linewidth=4, alpha=0.8,
                       label=f'Trend: {z[0]:.1f} days/°C')

        ax_climate.set_xlabel('Spring Temperature (°C)', fontsize=15, fontweight='bold')
        ax_climate.set_ylabel('Bloom Day of Year', fontsize=15, fontweight='bold')
        ax_climate.set_title('Toronto: Warmer Springs = Earlier Blooms',
                           fontsize=16, fontweight='bold')
        ax_climate.legend(fontsize=13, loc='upper right')
        ax_climate.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax_climate)
        cbar.set_label('Year', fontsize=13, fontweight='bold')

    # ========================================================================
    # BOTTOM MIDDLE: Toronto Feature Importance
    # ========================================================================
    ax_features = fig.add_subplot(gs[2, 1])

    if 'spring_temp' in results_df.columns:
        features = ['Spring\\nTemp', 'Spring\\nGDD', 'Winter\\nChill', 'Spring\\nPrecip']
        importance = [
            abs(np.corrcoef(results_df['spring_temp'], actual_doy)[0,1]),
            abs(np.corrcoef(results_df['spring_gdd'], actual_doy)[0,1]),
            abs(np.corrcoef(results_df['winter_chill_days'], actual_doy)[0,1]),
            abs(np.corrcoef(results_df['spring_precip'], actual_doy)[0,1])
        ]

        colors_feat = [COLORS['enhanced'] if i > 0.5 else COLORS['baseline'] for i in importance]
        bars = ax_features.bar(features, importance, color=colors_feat,
                              alpha=0.8, edgecolor='black', linewidth=2)

        # Add value labels
        for bar, val in zip(bars, importance):
            height = bar.get_height()
            ax_features.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{val:.2f}',
                            ha='center', va='bottom', fontsize=13, fontweight='bold')

        ax_features.set_ylabel('Correlation Strength', fontsize=14, fontweight='bold')
        ax_features.set_title('Key Climate Drivers', fontsize=16, fontweight='bold')
        ax_features.set_ylim(0, 1)
        ax_features.grid(True, alpha=0.3, axis='y')

    # ========================================================================
    # BOTTOM RIGHT: Toronto Summary Box
    # ========================================================================
    ax_summary = fig.add_subplot(gs[2, 2])
    ax_summary.axis('off')

    summary_text = f"""
    🏙️ TORONTO PREDICTIONS

    📍 Location: High Park
    📅 Years: {int(years.min())}-{int(years.max())}
    📊 Records: {len(results_df)}

    🎯 RESULTS:
    • Baseline MAE: {mae_baseline:.1f} days
    • Enhanced MAE: {mae_enhanced:.1f} days
    • Improvement: {improvement_pct:.0f}%

    🌡️ KEY INSIGHT:
    Temperature is Toronto's
    dominant bloom driver

    📈 METHOD:
    TabPFN Foundation Model
    with climate features

    ✓ Real Predictions
    """

    ax_summary.text(0.1, 0.9, summary_text,
                   transform=ax_summary.transAxes,
                   fontsize=14, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=1',
                            facecolor=COLORS['highlight'],
                            edgecolor=COLORS['toronto'],
                            linewidth=3, alpha=0.3))

    # Add big title overlay
    fig.text(0.5, 0.97, 'TORONTO CHERRY BLOSSOM PREDICTION',
            ha='center', fontsize=26, fontweight='bold',
            color=COLORS['toronto'])

    plt.savefig('visuals/toronto_hero.png', dpi=300, bbox_inches='tight',
               facecolor='white')
    print("✓ Generated: visuals/toronto_hero.png")
    plt.close()

def main():
    print("="*70)
    print("Generating Visuals in Original 3x3 Style")
    print("Using REAL TabPFN Predictions")
    print("="*70)
    print()

    os.makedirs('visuals', exist_ok=True)

    try:
        create_toronto_hero_original_style()

        print()
        print("="*70)
        print("✓ Original-Style Visual Generated!")
        print("="*70)
        print()
        print("Generated: visuals/toronto_hero.png")
        print("• 3x3 grid layout (EXACT original style)")
        print("• Cold/warm year annotations")
        print("• Real TabPFN predictions with uncertainties")
        print("• Climate correlations and feature importance")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run the comparison first:")
        print("  python3 compare_baseline_vs_climate.py")

if __name__ == '__main__':
    main()
