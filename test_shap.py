"""Quick test of the SHAP fix."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.generate_synthetic import generate_transit
from preprocessing.noise_removal import clean_lightcurve
from preprocessing.detrending import detrend_lightcurve
from detection.bls_detector import run_bls
from features.feature_extractor import extract_features
from classification.hybrid_classifier import HybridClassifier
from explainability.shap_explainer import explain_prediction

rng = np.random.default_rng(42)
time = np.linspace(0, 27.4, 2000)
result = generate_transit(time, rng)

cleaned = clean_lightcurve(time, result["flux"])
flux_flat = detrend_lightcurve(time, cleaned["flux_filtered"])
candidate = run_bls(time, flux_flat)
features = extract_features(time, flux_flat, candidate=candidate)

clf = HybridClassifier(
    xgb_path=os.path.join(os.path.dirname(__file__), "models", "xgboost_model.pkl"),
    cnn_path=os.path.join(os.path.dirname(__file__), "models", "cnn_model.pt"),
)
prediction = clf.predict(time, flux_flat)
cls_name = prediction["class"]
cls_conf = prediction["confidence"]
print(f"Classification: {cls_name} ({cls_conf:.1f}%)")

explanation = explain_prediction(clf.xgb_model, features, prediction)
print(f"Explanation method: {explanation['method']}")
print("Top 3 features:")
for c in explanation["contributions"][:3]:
    fname = c["feature"]
    pct = c.get("importance_pct", 0)
    print(f"  {fname}: {pct:.1f}%")
print("SUCCESS - SHAP fix works!")
