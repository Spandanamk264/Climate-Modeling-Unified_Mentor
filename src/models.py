from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import numpy as np

class ClimateModel:
    """
    Wrapper for multiple ML models used in climate change prediction.
    Supports: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
    """
    SUPPORTED_MODELS = {
        'linear': 'Linear Regression',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
        'xgboost': 'XGBoost',
    }

    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'decision_tree':
            self.model = DecisionTreeRegressor(max_depth=10, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported: {list(self.SUPPORTED_MODELS.keys())}")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        residuals = y_test.values - predictions if hasattr(y_test, 'values') else np.array(y_test) - predictions
        return {
            'MAE': mae,
            'MSE': mse,
            'R2': r2,
            'predictions': predictions,
            'residuals': residuals
        }

    def cross_validate(self, X, y, cv=5):
        """Performs k-fold cross-validation and returns scores."""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        return {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }


def compare_models(X_train, y_train, X_test, y_test):
    """
    Trains all supported models, evaluates them, and returns a comparison dict.
    """
    results = {}
    for key, name in ClimateModel.SUPPORTED_MODELS.items():
        model = ClimateModel(model_type=key)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        cv = model.cross_validate(X_train, y_train)
        results[name] = {
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'R2': metrics['R2'],
            'CV_Mean_R2': cv['cv_mean'],
            'CV_Std_R2': cv['cv_std'],
            'predictions': metrics['predictions'],
            'residuals': metrics['residuals'],
            'model_obj': model
        }
    return results
