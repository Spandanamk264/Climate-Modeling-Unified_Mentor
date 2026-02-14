from .data_generator import generate_synthetic_data
from .data_loader import load_data, preprocess_data
from .models import ClimateModel
from .visualization import plot_actual_vs_predicted, plot_feature_importance, plot_trends
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Starting Climate Change Modeling Project...")
    
    # Paths
    data_path = os.path.join("data", "climate_data.csv")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Data Generation (if not exists)
    if not os.path.exists(data_path):
        print("Generating synthetic dataset...")
        generate_synthetic_data(data_path)
    
    # 2. Load Data
    print("Loading data...")
    df = load_data(data_path)
    
    # 2b. Visualizing Trends
    print("Plotting trends...")
    plot_trends(df, ['Global_Temp_Anomaly', 'CO2_Concentration', 'Solar_Irradiance'], 
                save_path=os.path.join(results_dir, "trends.png"))

    # 3. Preprocessing
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df, target_column='Global_Temp_Anomaly')
    
    # 4. Model Training (Random Forest)
    print("Training Random Forest model...")
    rf_model = ClimateModel(model_type='random_forest')
    rf_model.train(X_train, y_train)
    
    # 5. Evaluation
    print("Evaluating model...")
    metrics = rf_model.evaluate(X_test, y_test)
    print(f"Metrics:\n MAE: {metrics['MAE']:.4f}\n MSE: {metrics['MSE']:.4f}\n R2: {metrics['R2']:.4f}")
    
    # 6. Visualization
    y_pred = metrics['predictions']
    plot_actual_vs_predicted(y_test, y_pred, "Actual vs Predicted Temperature Anomaly", 
                             save_path=os.path.join(results_dir, "prediction_accuracy.png"))
    
    plot_feature_importance(rf_model.model, feature_names, 
                            save_path=os.path.join(results_dir, "feature_importance.png"))

    print(f"Project completed successfully. Results saved in '{results_dir}'.")

if __name__ == "__main__":
    main()
