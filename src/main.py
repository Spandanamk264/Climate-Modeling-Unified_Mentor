from .data_generator import generate_synthetic_data
from .data_loader import load_data, preprocess_data
from .models import ClimateModel, compare_models
from .visualization import plot_actual_vs_predicted, plot_feature_importance, plot_trends
import os
import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("   Climate Change Modeling - Advanced ML Pipeline")
    print("=" * 60)

    # Paths
    data_path = os.path.join("data", "climate_data.csv")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Data Generation
    if not os.path.exists(data_path):
        print("\n[Step 1] Generating synthetic dataset...")
        generate_synthetic_data(data_path)
    else:
        print("\n[Step 1] Dataset already exists.")

    # 2. Load Data
    print("[Step 2] Loading data...")
    df = load_data(data_path)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    # 3. Visualize Trends
    print("[Step 3] Plotting trends...")
    plot_trends(df, ['Global_Temp_Anomaly', 'CO2_Concentration', 'Solar_Irradiance'],
                save_path=os.path.join(results_dir, "trends.png"))

    # 4. Preprocessing (with Feature Engineering)
    print("[Step 4] Preprocessing with feature engineering...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df, target_column='Global_Temp_Anomaly')
    print(f"   Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"   Features: {list(feature_names)}")

    # 5. Model Comparison (All 5 models)
    print("[Step 5] Training and comparing ALL models...")
    results = compare_models(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("   MODEL COMPARISON RESULTS")
    print("=" * 60)
    best_r2 = -1
    best_model_name = ""
    for name, res in results.items():
        print(f"\n   {name}:")
        print(f"      R2: {res['R2']:.4f}  |  MAE: {res['MAE']:.4f}  |  MSE: {res['MSE']:.4f}")
        print(f"      CV R2: {res['CV_Mean_R2']:.4f} +/- {res['CV_Std_R2']:.4f}")
        if res['R2'] > best_r2:
            best_r2 = res['R2']
            best_model_name = name

    print(f"\n   BEST MODEL: {best_model_name} (R2 = {best_r2:.4f})")
    print("=" * 60)

    # 6. Visualization (Best Model)
    best_result = results[best_model_name]
    plot_actual_vs_predicted(y_test, best_result['predictions'], f"Actual vs Predicted ({best_model_name})",
                             save_path=os.path.join(results_dir, "prediction_accuracy.png"))

    best_model_obj = best_result['model_obj'].model
    if hasattr(best_model_obj, 'feature_importances_'):
        plot_feature_importance(best_model_obj, feature_names,
                                save_path=os.path.join(results_dir, "feature_importance.png"))

    print(f"\nProject completed. Results saved in '{results_dir}'.")

if __name__ == "__main__":
    main()
