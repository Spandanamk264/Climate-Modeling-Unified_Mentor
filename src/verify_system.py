import os
import sys
import pandas as pd
import numpy as np
from data_generator import generate_synthetic_data
from data_loader import load_data, preprocess_data
from models import ClimateModel

def test_system():
    print("🔍 Starting System Verification...")
    
    # 1. Test Data Generation
    data_path = os.path.join("data", "test_climate_data.csv")
    print(f"   [1/4] Testing Data Generation at {data_path}...")
    try:
        df = generate_synthetic_data(data_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError("Data file was not created.")
        if df.empty:
            raise ValueError("Generated dataframe is empty.")
        print("   ✅ Data Generation Passed.")
    except Exception as e:
        print(f"   ❌ Data Generation Failed: {e}")
        return

    # 2. Test Data Loading & Preprocessing
    print("   [2/4] Testing Data Loading & Preprocessing...")
    try:
        df_loaded = load_data(data_path)
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df_loaded, target_column='Global_Temp_Anomaly')
        
        if X_train.shape[0] == 0:
            raise ValueError("Training set is empty.")
        print("   ✅ Preprocessing Passed.")
    except Exception as e:
        print(f"   ❌ Preprocessing Failed: {e}")
        return

    # 3. Test Model Training (Random Forest)
    print("   [3/4] Testing Random Forest Model Training...")
    try:
        model = ClimateModel(model_type='random_forest')
        model.train(X_train, y_train)
        print("   ✅ Model Training Passed.")
    except Exception as e:
        print(f"   ❌ Model Training Failed: {e}")
        return

    # 4. Test Prediction & Evaluation
    print("   [4/4] Testing Prediction & Evaluation...")
    try:
        metrics = model.evaluate(X_test, y_test)
        print(f"      -> R2 Score: {metrics['R2']:.4f}")
        
        # Test Single Prediction (Simulation)
        sample_input = np.array([[420.0, 1361.0, 0.0, 100.0]]) # CO2, Solar, Volcanic, SeaLevel
        # Note: We need to ensure the scaler works on this input shape. 
        # The scaler was fitted on X which has 4 columns (CO2, Solar, Volcanic, Sea_Level).
        # Let's verify feature count.
        if sample_input.shape[1] != X_train.shape[1]:
             print(f"      ⚠️ Warning: Input shape mismatch. Model expects {X_train.shape[1]} features, got {sample_input.shape[1]}.")
        
        sample_scaled = scaler.transform(sample_input)
        prediction = model.predict(sample_scaled)
        print(f"      -> Test Prediction: {prediction[0]:.4f}")
        
        print("   ✅ Prediction Passed.")
    except Exception as e:
        print(f"   ❌ Prediction Failed: {e}")
        return

    print("\n🎉 ALL SYSTEMS GO! The backend logic is functioning correctly.")
    
    # Cleanup
    if os.path.exists(data_path):
        os.remove(data_path)
        print("   (Test data cleaned up)")

if __name__ == "__main__":
    test_system()
