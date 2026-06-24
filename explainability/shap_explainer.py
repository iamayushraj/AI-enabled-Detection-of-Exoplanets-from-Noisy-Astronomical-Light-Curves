"""
SHAP Explainability Module for ExoplanetAI

Provides feature importance explanations for XGBoost predictions
using SHAP (SHapley Additive exPlanations).
"""

import numpy as np
import os

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.feature_extractor import LightCurveFeatures


CLASS_NAMES = ["transit", "eclipsing_binary", "variable_star", "blend", "noise"]


def explain_prediction(model, features: LightCurveFeatures,
                       prediction: dict) -> dict:
    """
    Generate SHAP explanation for a single prediction.

    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost model.
    features : LightCurveFeatures
        Extracted features.
    prediction : dict
        Prediction result from classifier.

    Returns
    -------
    explanation : dict
        Feature contributions and importance data.
    """
    feature_array = features.to_array().reshape(1, -1)
    feature_names = LightCurveFeatures.feature_names()
    predicted_class = prediction.get("class", "unknown")
    class_id = prediction.get("class_id", 0)

    if HAS_SHAP:
        return _shap_explain(model, feature_array, feature_names,
                             predicted_class, class_id)
    else:
        return _fallback_explain(model, feature_array, feature_names,
                                 predicted_class, class_id)


def _shap_explain(model, feature_array, feature_names,
                  predicted_class, class_id) -> dict:
    """SHAP-based explanation with robust shape handling."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_array)

        # Handle different SHAP output formats:
        # - list of arrays: shap_values[class_id] shape (n_samples, n_features)
        # - 3D array: shape (n_samples, n_features, n_classes)
        # - 2D array: shape (n_samples, n_features) for binary
        if isinstance(shap_values, list):
            class_shap = shap_values[class_id][0]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # (n_samples, n_features, n_classes)
                class_shap = shap_values[0, :, class_id]
            elif shap_values.ndim == 2:
                class_shap = shap_values[0]
            else:
                class_shap = shap_values.flatten()[:len(feature_names)]
        else:
            # shap Explanation object
            sv = shap_values.values
            if sv.ndim == 3:
                class_shap = sv[0, :, class_id]
            elif sv.ndim == 2:
                class_shap = sv[0]
            else:
                class_shap = sv.flatten()[:len(feature_names)]

        # Sort by absolute importance
        sorted_idx = np.argsort(np.abs(class_shap))[::-1]

        contributions = []
        for idx in sorted_idx:
            contributions.append({
                "feature": feature_names[idx],
                "value": round(float(feature_array[0, idx]), 6),
                "shap_value": round(float(class_shap[idx]), 6),
                "impact": "positive" if class_shap[idx] > 0 else "negative",
                "importance_pct": round(abs(float(class_shap[idx])) /
                                         (np.sum(np.abs(class_shap)) + 1e-10) * 100, 1),
            })

        # Natural language explanation
        top_3 = contributions[:3]
        explanation_text = _generate_text_explanation(predicted_class, top_3)

        return {
            "predicted_class": predicted_class,
            "contributions": contributions,
            "explanation_text": explanation_text,
            "method": "SHAP TreeExplainer",
        }
    except Exception as e:
        # Fall back to feature importance if SHAP fails
        print(f"SHAP failed ({e}), falling back to feature importance")
        return _fallback_explain(model, feature_array, feature_names,
                                 predicted_class, class_id)


def _fallback_explain(model, feature_array, feature_names,
                       predicted_class, class_id) -> dict:
    """Fallback explanation using feature importances when SHAP is unavailable."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        importances = np.ones(len(feature_names)) / len(feature_names)

    sorted_idx = np.argsort(importances)[::-1]

    contributions = []
    for idx in sorted_idx:
        contributions.append({
            "feature": feature_names[idx],
            "value": round(float(feature_array[0, idx]), 6),
            "importance": round(float(importances[idx]), 6),
            "importance_pct": round(float(importances[idx]) /
                                     (importances.sum() + 1e-10) * 100, 1),
        })

    top_3 = contributions[:3]
    explanation_text = _generate_text_explanation(predicted_class, top_3)

    return {
        "predicted_class": predicted_class,
        "contributions": contributions,
        "explanation_text": explanation_text,
        "method": "Feature Importance (SHAP unavailable)",
    }


def _generate_text_explanation(predicted_class: str, top_features: list) -> str:
    """Generate a human-readable explanation."""
    class_descriptions = {
        "transit": "an exoplanet transit (a planet crossing in front of its host star)",
        "eclipsing_binary": "an eclipsing binary system (two stars orbiting each other)",
        "variable_star": "intrinsic stellar variability (pulsation or activity)",
        "blend": "a blended signal (contaminated by nearby sources)",
        "noise": "instrumental noise or artifacts (no astrophysical signal)",
    }

    feature_descriptions = {
        "transit_depth": "transit depth (brightness decrease)",
        "transit_duration": "transit duration",
        "orbital_period": "orbital period",
        "snr": "signal-to-noise ratio",
        "transit_symmetry": "transit shape symmetry",
        "n_transits": "number of detected transits",
        "bls_power": "BLS detection power",
        "sde": "signal detection efficiency",
        "depth_even_odd": "even/odd transit depth ratio",
        "secondary_depth": "secondary eclipse depth",
        "flux_std": "flux variability",
        "flux_skewness": "flux distribution asymmetry",
        "flux_kurtosis": "flux distribution shape",
        "amplitude": "overall brightness amplitude",
    }

    desc = class_descriptions.get(predicted_class, f"class '{predicted_class}'")
    lines = [f"This signal was classified as {desc}."]
    lines.append("")
    lines.append("The top contributing factors were:")

    for i, feat in enumerate(top_features, 1):
        name = feat["feature"]
        pct = feat.get("importance_pct", feat.get("shap_value", 0))
        desc_f = feature_descriptions.get(name, name)
        lines.append(f"  {i}. {desc_f} — contributed {pct:.1f}% to this classification")

    return "\n".join(lines)


def get_global_importance(model) -> dict:
    """Get global feature importance from the model."""
    feature_names = LightCurveFeatures.feature_names()
    try:
        importances = model.feature_importances_
    except AttributeError:
        return {}

    sorted_idx = np.argsort(importances)[::-1]
    return {
        feature_names[i]: round(float(importances[i]), 6)
        for i in sorted_idx
    }
