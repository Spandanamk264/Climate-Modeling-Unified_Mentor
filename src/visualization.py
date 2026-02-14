import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_actual_vs_predicted(y_test, y_pred, title, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, save_path=None):
    if not hasattr(model, 'feature_importances_'):
        print("Model does not calculate feature importance.")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_trends(df, columns, save_path=None):
    plt.figure(figsize=(12, 6))
    for col in columns:
        # Normalize for plotting on same scale
        normalized = (df[col] - df[col].mean()) / df[col].std()
        plt.plot(df['Year'], normalized, label=col)
    
    plt.xlabel('Year')
    plt.ylabel('Normalized Value')
    plt.title('Climate Trends Over Time')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()
