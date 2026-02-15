import os
import sys
import numpy as np
from data_generator import generate_synthetic_data
from data_loader import load_data, preprocess_data, feature_engineering
from models import ClimateModel, compare_models

def test_system():
    print("Starting System Verification...")

    data_path = os.path.join("data", "test_climate_data.csv")

    # 1. Data Generation
    print("   [1/6] Testing Data Generation...")
    try:
        df = generate_synthetic_data(data_path)
        assert os.path.exists(data_path), "File not created"
        assert not df.empty, "DataFrame is empty"
        assert 'Precipitation' in df.columns, "Missing Precipitation column"
        print("   PASS: Data Generation")
    except Exception as e:
        print(f"   FAIL: Data Generation - {e}")
        return

    # 2. Feature Engineering
    print("   [2/6] Testing Feature Engineering...")
    try:
        fe_df = feature_engineering(df)
        assert 'CO2_Rolling_3yr' in fe_df.columns, "Missing rolling average"
        assert 'CO2_Lag_1yr' in fe_df.columns, "Missing lag variable"
        assert 'CO2_Rate_of_Change' in fe_df.columns, "Missing rate of change"
        print("   PASS: Feature Engineering")
    except Exception as e:
        print(f"   FAIL: Feature Engineering - {e}")
        return

    # 3. Preprocessing
    print("   [3/6] Testing Preprocessing...")
    try:
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
        assert X_train.shape[0] > 0, "Empty training set"
        assert len(feature_names) > 5, "Too few features after engineering"
        print(f"   PASS: Preprocessing ({len(feature_names)} features)")
    except Exception as e:
        print(f"   FAIL: Preprocessing - {e}")
        return

    # 4. All Models Training
    print("   [4/6] Testing All 5 Models...")
    try:
        for mtype in ['linear', 'decision_tree', 'random_forest', 'gradient_boosting', 'xgboost']:
            m = ClimateModel(mtype)
            m.train(X_train, y_train)
            res = m.evaluate(X_test, y_test)
            print(f"      {mtype}: R2={res['R2']:.4f}")
        print("   PASS: All Models")
    except Exception as e:
        print(f"   FAIL: Model Training - {e}")
        return

    # 5. Cross-Validation
    print("   [5/6] Testing Cross-Validation...")
    try:
        m = ClimateModel('random_forest')
        m.train(X_train, y_train)
        cv = m.cross_validate(X_train, y_train, cv=5)
        assert 'cv_mean' in cv, "Missing cv_mean"
        print(f"   PASS: Cross-Validation (CV R2={cv['cv_mean']:.4f})")
    except Exception as e:
        print(f"   FAIL: Cross-Validation - {e}")
        return

    # 6. Model Comparison
    print("   [6/6] Testing Model Comparison...")
    try:
        results = compare_models(X_train, y_train, X_test, y_test)
        assert len(results) == 5, f"Expected 5 models, got {len(results)}"
        print("   PASS: Model Comparison")
    except Exception as e:
        print(f"   FAIL: Model Comparison - {e}")
        return

    print("\nALL 6 TESTS PASSED! Project is fully functional.")

    # Cleanup
    if os.path.exists(data_path):
        os.remove(data_path)
        print("   (Test data cleaned up)")

if __name__ == "__main__":
    test_system()
