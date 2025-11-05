"""
Climate Feature Engineering for Cherry Blossom Prediction

This script demonstrates how to augment the basic features (lat, long, alt, year)
with climate data that could significantly improve predictions.

Data Sources:
- Environment Canada Climate Data: https://climate.weather.gc.ca/
- Open-Meteo API (free, no auth required): https://open-meteo.com/
- Visual Crossing (commercial): https://www.visualcrossing.com/
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time

def get_open_meteo_historical(lat, lon, start_date, end_date):
    """
    Fetch historical weather data from Open-Meteo API (free, no API key needed)

    Parameters:
    - lat, lon: Location coordinates
    - start_date, end_date: Date range (YYYY-MM-DD format)

    Returns:
    - DataFrame with daily weather data
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "snowfall_sum",
            "rain_sum"
        ],
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'temp_max': data['daily']['temperature_2m_max'],
            'temp_min': data['daily']['temperature_2m_min'],
            'temp_mean': data['daily']['temperature_2m_mean'],
            'precipitation': data['daily']['precipitation_sum'],
            'snowfall': data['daily']['snowfall_sum'],
            'rain': data['daily']['rain_sum']
        })

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def calculate_growing_degree_days(temp_mean, base_temp=5.0):
    """
    Calculate Growing Degree Days (GDD)
    GDD = max(0, temp_mean - base_temp)

    Common base temperatures:
    - 5°C for many plants
    - 10°C for some crops
    """
    return max(0, temp_mean - base_temp)


def engineer_climate_features(df_weather, bloom_year):
    """
    Engineer climate features relevant to cherry blossom prediction

    Focus on winter/spring period before bloom (typically Dec-Apr)
    """
    features = {}

    # Filter to relevant period (previous December through April of bloom year)
    start_winter = f"{bloom_year-1}-12-01"
    end_spring = f"{bloom_year}-04-30"

    period = df_weather[
        (df_weather['date'] >= start_winter) &
        (df_weather['date'] <= end_spring)
    ].copy()

    if len(period) == 0:
        return None

    # Temperature features
    features['winter_temp_mean'] = period[
        (period['date'] >= f"{bloom_year-1}-12-01") &
        (period['date'] <= f"{bloom_year}-02-28")
    ]['temp_mean'].mean()

    features['spring_temp_mean'] = period[
        (period['date'] >= f"{bloom_year}-03-01") &
        (period['date'] <= f"{bloom_year}-04-30")
    ]['temp_mean'].mean()

    features['jan_temp_mean'] = period[
        period['date'].dt.month == 1
    ]['temp_mean'].mean()

    features['feb_temp_mean'] = period[
        period['date'].dt.month == 2
    ]['temp_mean'].mean()

    features['mar_temp_mean'] = period[
        period['date'].dt.month == 3
    ]['temp_mean'].mean()

    # Growing Degree Days (cumulative)
    period['gdd'] = period['temp_mean'].apply(lambda x: calculate_growing_degree_days(x, base_temp=5.0))
    features['gdd_cumulative'] = period['gdd'].sum()

    # Precipitation features
    features['winter_precip_total'] = period[
        (period['date'] >= f"{bloom_year-1}-12-01") &
        (period['date'] <= f"{bloom_year}-02-28")
    ]['precipitation'].sum()

    features['spring_precip_total'] = period[
        (period['date'] >= f"{bloom_year}-03-01") &
        (period['date'] <= f"{bloom_year}-04-30")
    ]['precipitation'].sum()

    # Snow features
    features['total_snowfall'] = period['snowfall'].sum()

    # Frost days (days with min temp below 0°C)
    features['frost_days'] = (period['temp_min'] < 0).sum()

    # First spring day (first day with mean temp > 10°C in spring)
    spring_warm = period[
        (period['date'] >= f"{bloom_year}-03-01") &
        (period['temp_mean'] > 10)
    ]
    if len(spring_warm) > 0:
        first_warm_day = spring_warm['date'].min()
        features['first_warm_day_doy'] = first_warm_day.timetuple().tm_yday
    else:
        features['first_warm_day_doy'] = None

    return features


def demonstrate_feature_engineering():
    """
    Demonstrate climate feature engineering for Toronto
    """
    print("="*70)
    print("CLIMATE FEATURE ENGINEERING DEMONSTRATION")
    print("="*70)

    # Toronto coordinates
    toronto_lat = 43.646548
    toronto_lon = -79.463690

    # Example: Fetch weather data for 2023 bloom year
    bloom_year = 2023
    start_date = f"{bloom_year-1}-12-01"
    end_date = f"{bloom_year}-04-30"

    print(f"\nFetching weather data for Toronto...")
    print(f"Period: {start_date} to {end_date}")
    print(f"Location: {toronto_lat}°N, {toronto_lon}°W")

    df_weather = get_open_meteo_historical(toronto_lat, toronto_lon, start_date, end_date)

    if df_weather is not None:
        print(f"\n✓ Retrieved {len(df_weather)} days of weather data")
        print(f"\nSample data:")
        print(df_weather.head(10).to_string(index=False))

        # Engineer features
        print(f"\n" + "="*70)
        print("ENGINEERED CLIMATE FEATURES")
        print("="*70)

        features = engineer_climate_features(df_weather, bloom_year)

        if features:
            print(f"\nFeatures for {bloom_year} bloom prediction:")
            print("-"*70)
            for key, value in features.items():
                if value is not None:
                    if 'temp' in key or 'gdd' in key:
                        print(f"  {key:<25} {value:>10.2f} °C")
                    elif 'precip' in key or 'snowfall' in key:
                        print(f"  {key:<25} {value:>10.2f} mm")
                    elif 'days' in key or 'doy' in key:
                        print(f"  {key:<25} {value:>10.0f}")
                    else:
                        print(f"  {key:<25} {value:>10}")

        # Show summary statistics
        print(f"\n" + "="*70)
        print("WEATHER SUMMARY STATISTICS")
        print("="*70)
        print(df_weather[['temp_mean', 'precipitation', 'snowfall']].describe())

    else:
        print("\n✗ Failed to fetch weather data")

    print(f"\n" + "="*70)
    print("NOTES")
    print("="*70)
    print("""
These climate features could significantly improve bloom predictions:

Key Features:
1. Winter/Spring Temperature Means - Directly affect bloom timing
2. Growing Degree Days (GDD) - Index of accumulated heat
3. First Warm Day - Indicator of spring onset
4. Frost Days - Can delay or damage blooms
5. Precipitation/Snowfall - Affect soil moisture and timing

To augment the full dataset:
1. Fetch historical weather for ALL locations (331 locations)
2. Engineer features for each year in the dataset
3. Add to existing lat/long/alt/year features
4. Retrain TabPFN with enriched features

Expected Improvement:
- Current baseline (lat/long/alt/year): MAE ~7-12 days
- With climate features: MAE potentially ~3-7 days
""")


if __name__ == "__main__":
    demonstrate_feature_engineering()
