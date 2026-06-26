import os
import sys
import pickle
import numpy as np

# Load local modules
from classification.xgboost_classifier import load_model
from features.feature_extractor import LightCurveFeatures
from explainability.shap_explainer import explain_prediction

def test_shap():
    print("Loading XGBoost model...")
    model_path = os.path.join("models", "xgboost_model.pkl")
    model = load_model(model_path)
    
    print("Creating dummy features...")
    # 11 features corresponding to LightCurveFeatures
    dummy_feat_array = np.array([0.01, 1.5, 0.005, 5.0, 3.2, 0.1, 0.8, 0.2, 0.05, 0.9, 0.0])
    
    # Create dummy LightCurveFeatures
    features = LightCurveFeatures(
        transit_depth=0.01,
        duration=1.5,
        secondary_depth=0.005,
        period=5.0,
        sde=3.2,
        amplitude=0.1,
        mad=0.8,
        residual_rms=0.2,
        snr=0.05,
        autocorr_lag1=0.9,
        centroid_offset=0.0
    )
    
    print("Predicting...")
    # XGBoost classifier predict
    prediction = {
        "class": "transit",
        "class_id": 0,
    }
    
    print("Explaining prediction with SHAP...")
    explanation = explain_prediction(model, features, prediction)
    
    print("Explanation result:")
    print("Method:", explanation.get("method"))
    for c in explanation.get("contributions", [])[:3]:
        print(f"Feature: {c['feature']}, Impact: {c['impact']}, Importance: {c['importance_pct']}%")

if __name__ == "__main__":
    test_shap()
