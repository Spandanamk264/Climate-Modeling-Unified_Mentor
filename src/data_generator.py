import pandas as pd
import numpy as np
import os

def generate_synthetic_data(filepath):
    """
    Generates a synthetic climate dataset and saves it to a CSV file.
    Columns: Year, CO2_Concentration, Solar_Irradiance, Volcanic_Activity,
             Precipitation, Global_Temp_Anomaly, Sea_Level_Rise
    """
    np.random.seed(42)
    years = np.arange(1900, 2024)
    n = len(years)

    # CO2: Exponential increase (baseline ~295ppm in 1900, ~420ppm in 2023)
    co2 = 295 + (years - 1900)**1.8 * 0.015 + np.random.normal(0, 1, n)

    # Solar: 11-year cycle
    solar = 1361 + 0.5 * np.sin(2 * np.pi * (years - 1900) / 11) + np.random.normal(0, 0.1, n)

    # Volcanic: Sparse spikes (negative forcing, cooling effect)
    volcanic = np.zeros(n)
    volcanic_indices = np.random.choice(n, 5, replace=False)
    volcanic[volcanic_indices] = -0.3 + np.random.normal(0, 0.1, 5)

    # Precipitation: Slightly increasing with noise
    precipitation = 1000 + 0.5 * (years - 1900) + np.random.normal(0, 30, n)

    # Temperature Anomaly: Dependent on CO2, Solar, Volcanic
    temp_anomaly = -0.5 + 0.01 * (co2 - 295) + 0.1 * (solar - 1361) + volcanic + np.random.normal(0, 0.1, n)

    # Sea Level Rise: Cumulative integral of temperature (simplified)
    sea_level_rate = 1.5 + 2.0 * (temp_anomaly - (-0.5))  # mm/year
    sea_level = np.cumsum(sea_level_rate)
    sea_level = sea_level - sea_level[0]  # Start at 0 relative to 1900

    data = pd.DataFrame({
        'Year': years,
        'CO2_Concentration': co2,
        'Solar_Irradiance': solar,
        'Volcanic_Activity': volcanic,
        'Precipitation': precipitation,
        'Global_Temp_Anomaly': temp_anomaly,
        'Sea_Level_Rise': sea_level
    })

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath, index=False)
    print(f"Synthetic data generated at {filepath}")
    return data

if __name__ == "__main__":
    generate_synthetic_data(os.path.join("data", "climate_data.csv"))
