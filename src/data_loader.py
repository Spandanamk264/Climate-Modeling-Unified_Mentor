import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Benchmarks loading the dataset."""
    if not isinstance(filepath, str):
        raise ValueError("Filepath must be a string")
    return pd.read_csv(filepath)

def preprocess_data(df, target_column='Global_Temp_Anomaly', test_size=0.2, random_state=42):
    """
    Preprocesses the data: handles missing values (imputation), splits features/target, 
    splits train/test, and scales features.
    """
    # 1. Handle missing values (simple fill for this mock data, or drop)
    df = df.dropna()

    # 2. Separate features and target
    X = df.drop(columns=['Year', target_column]) # Year is usually not a feature for causal models unless time-series specific
    y = df[target_column]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 4. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def prepare_future_data(last_year_data, years_to_predict=10):
    """
    Prepares future feature data based on simple extrapolation for scenarios.
    """
    # This is a placeholder for more complex scenario generation
    # We will just repeat the last known values with increments
    # In a real app, this would be based on RCP scenarios
    pass
