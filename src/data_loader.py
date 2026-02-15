import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Loads the climate dataset from a CSV file."""
    if not isinstance(filepath, str):
        raise ValueError("Filepath must be a string")
    return pd.read_csv(filepath)

def feature_engineering(df):
    """
    Creates new features useful for prediction:
    - Rolling averages (3-year and 5-year) for CO2 and Temperature
    - Lagged variables (1-year and 2-year lag) for CO2 and Temperature
    - Rate of change for CO2
    """
    df = df.copy()
    df = df.sort_values('Year').reset_index(drop=True)

    # Rolling Averages
    df['CO2_Rolling_3yr'] = df['CO2_Concentration'].rolling(window=3, min_periods=1).mean()
    df['CO2_Rolling_5yr'] = df['CO2_Concentration'].rolling(window=5, min_periods=1).mean()
    df['Temp_Rolling_3yr'] = df['Global_Temp_Anomaly'].rolling(window=3, min_periods=1).mean()

    # Lagged Variables
    df['CO2_Lag_1yr'] = df['CO2_Concentration'].shift(1)
    df['CO2_Lag_2yr'] = df['CO2_Concentration'].shift(2)
    df['Temp_Lag_1yr'] = df['Global_Temp_Anomaly'].shift(1)

    # Rate of Change
    df['CO2_Rate_of_Change'] = df['CO2_Concentration'].diff()

    # Drop NaN rows created by shift/rolling (first few rows)
    df = df.dropna().reset_index(drop=True)

    return df

def preprocess_data(df, target_column='Global_Temp_Anomaly', test_size=0.2, random_state=42, use_feature_engineering=True):
    """
    Preprocesses the data:
    1. Applies feature engineering (rolling averages, lags)
    2. Handles missing values
    3. Splits features/target
    4. Splits train/test
    5. Standardizes features
    """
    if use_feature_engineering:
        df = feature_engineering(df)

    # Handle missing values
    df = df.dropna()

    # Separate features and target
    drop_cols = ['Year', target_column]
    # Also drop Temp rolling/lag if predicting temp (to avoid target leakage)
    leakage_cols = [c for c in df.columns if 'Temp_Rolling' in c or 'Temp_Lag' in c]
    drop_cols.extend(leakage_cols)
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_column]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns
