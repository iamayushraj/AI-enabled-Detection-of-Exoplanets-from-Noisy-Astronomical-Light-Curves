"""
Hybrid Classifier for ExoplanetAI

Combines BLS detection → Feature Extraction → XGBoost + CNN ensemble
for robust transit classification with calibrated confidence scores.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detection.bls_detector import run_bls, phase_fold
from features.feature_extractor import extract_features
from classification.xgboost_classifier import (
    predict_xgboost, load_model as load_xgb,
    CLASS_NAMES
)
from classification.cnn_classifier import (
    predict_cnn, load_cnn, prepare_phase_folded_input
)


class HybridClassifier:
    """
    Ensemble classifier combining XGBoost (features) + CNN (phase-folded curve).

    Pipeline:
        1. Run BLS to detect transit candidate
        2. Extract features → XGBoost prediction
        3. Phase-fold on BLS period → CNN prediction
        4. Ensemble: weighted average of probabilities
    """

    def __init__(self, xgb_path: str = None, cnn_path: str = None,
                 xgb_weight: float = 0.6, cnn_weight: float = 0.4,
                 device: str = "cpu"):
        """
        Parameters
        ----------
        xgb_path : str
            Path to saved XGBoost model (.pkl).
        cnn_path : str
            Path to saved CNN model (.pt).
        xgb_weight : float
            Weight for XGBoost in ensemble (0-1).
        cnn_weight : float
            Weight for CNN in ensemble (0-1).
        device : str
            PyTorch device for CNN.
        """
        self.xgb_model = None
        self.cnn_model = None
        self.xgb_weight = xgb_weight
        self.cnn_weight = cnn_weight
        self.device = device

        if xgb_path and os.path.exists(xgb_path):
            self.xgb_model = load_xgb(xgb_path)
            print(f"XGBoost model loaded from {xgb_path}")

        if cnn_path and os.path.exists(cnn_path):
            self.cnn_model = load_cnn(cnn_path, device=device)
            print(f"CNN model loaded from {cnn_path}")

    def predict(self, time: np.ndarray, flux: np.ndarray) -> dict:
        """
        Full pipeline prediction.

        Parameters
        ----------
        time : np.ndarray
            Time array (days).
        flux : np.ndarray
            Cleaned, detrended flux.

        Returns
        -------
        result : dict
            Full prediction with class, confidence, parameters, and individual model outputs.
        """
        # Step 1: BLS detection
        candidate = run_bls(time, flux)

        # Step 2: Feature extraction
        features = extract_features(time, flux, candidate=candidate)

        # Step 3: Get predictions from available models
        predictions = {}
        ensemble_probs = np.zeros(len(CLASS_NAMES))
        total_weight = 0.0

        # XGBoost prediction
        if self.xgb_model is not None:
            xgb_pred = predict_xgboost(self.xgb_model, features.to_array())
            predictions["xgboost"] = xgb_pred
            xgb_probs = np.array([
                xgb_pred["probabilities"].get(c, 0.0) / 100.0 for c in CLASS_NAMES
            ])
            ensemble_probs += self.xgb_weight * xgb_probs
            total_weight += self.xgb_weight

        # CNN prediction
        if self.cnn_model is not None and candidate is not None:
            phase_folded = prepare_phase_folded_input(
                time, flux, candidate.period, candidate.t0
            )
            cnn_pred = predict_cnn(self.cnn_model, phase_folded, self.device)
            predictions["cnn"] = cnn_pred
            cnn_probs = np.array([
                cnn_pred["probabilities"][c] / 100.0 for c in CLASS_NAMES
            ])
            ensemble_probs += self.cnn_weight * cnn_probs
            total_weight += self.cnn_weight

        # Normalize ensemble
        if total_weight > 0:
            ensemble_probs /= total_weight
        else:
            # Fallback: use features heuristic
            ensemble_probs = _heuristic_classify(features)

        class_id = int(np.argmax(ensemble_probs))

        return {
            "class": CLASS_NAMES[class_id],
            "class_id": class_id,
            "confidence": round(float(ensemble_probs[class_id]) * 100, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(float(p) * 100, 2)
                for i, p in enumerate(ensemble_probs)
            },
            "features": features.to_dict(),
            "transit_candidate": candidate.to_dict() if candidate else None,
            "individual_predictions": predictions,
        }

    def batch_predict(self, time_list: list, flux_list: list) -> list[dict]:
        """Run prediction on multiple light curves."""
        results = []
        for i, (t, f) in enumerate(zip(time_list, flux_list)):
            try:
                result = self.predict(t, f)
                result["index"] = i
                results.append(result)
            except Exception as e:
                results.append({
                    "index": i,
                    "class": "error",
                    "confidence": 0.0,
                    "error": str(e),
                })
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(time_list)} light curves")
        return results


def _heuristic_classify(features) -> np.ndarray:
    """Simple heuristic classification when no ML models are loaded."""
    probs = np.ones(5) * 0.1  # base probability

    f = features

    # Strong transit indicators
    if f.sde > 7 and f.transit_depth > 0.0005:
        probs[0] = 0.7  # transit
        if f.secondary_depth > f.transit_depth * 0.1:
            probs[1] = 0.5  # could be binary
            probs[0] = 0.3
    elif f.transit_depth > 0.03:
        probs[1] = 0.6  # likely binary (deep)
    elif f.amplitude > 0.01 and f.sde < 5:
        probs[2] = 0.6  # variable star
    elif f.sde < 4:
        probs[4] = 0.5  # noise

    # Normalize
    probs /= probs.sum()
    return probs
