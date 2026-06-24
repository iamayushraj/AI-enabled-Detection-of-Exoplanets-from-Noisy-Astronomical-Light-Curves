"""
XGBoost Classifier for ExoplanetAI

5-class light curve classification using extracted features:
  0: transit
  1: eclipsing_binary
  2: variable_star
  3: blend
  4: noise
"""

import numpy as np
import os
import pickle
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

CLASS_NAMES = ["transit", "eclipsing_binary", "variable_star", "blend", "noise"]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}


def build_xgboost_model(n_classes: int = 5, **kwargs) -> XGBClassifier:
    """
    Create an XGBoost classifier with tuned hyperparameters.
    """
    defaults = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": 1,
    }
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   n_folds: int = 5) -> dict:
    """
    Train XGBoost classifier with cross-validation.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (n_samples, n_features).
    y_train : np.ndarray
        Training labels (integer-encoded).
    X_val : np.ndarray, optional
        Validation features.
    y_val : np.ndarray, optional
        Validation labels.
    n_folds : int
        Number of CV folds.

    Returns
    -------
    result : dict
        Keys: 'model', 'cv_scores', 'cv_mean', 'val_report', 'confusion_matrix'
    """
    model = build_xgboost_model()

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring='accuracy', n_jobs=1)

    # Train on full training set
    model.fit(X_train, y_train)

    result = {
        "model": model,
        "cv_scores": cv_scores,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }

    # Evaluate on validation set
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        if y_pred.ndim > 1:
            y_pred = y_pred.argmax(axis=1)
            
        present_labels = np.unique(np.concatenate([y_val, y_pred]))
        target_names_present = [CLASS_NAMES[i] for i in present_labels]
        report = classification_report(y_val, y_pred,
                                        labels=present_labels,
                                        target_names=target_names_present,
                                        output_dict=True)
        cm = confusion_matrix(y_val, y_pred)
        result["val_report"] = report
        result["confusion_matrix"] = cm
        result["val_accuracy"] = float(report["accuracy"])

    return result


def predict_xgboost(model: XGBClassifier,
                     features: np.ndarray) -> dict:
    """
    Predict class and confidence for a single light curve.

    Parameters
    ----------
    model : XGBClassifier
        Trained model.
    features : np.ndarray
        Feature vector (1D) or matrix (2D).

    Returns
    -------
    prediction : dict
        Keys: 'class', 'class_id', 'confidence', 'probabilities'
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)

    probs = model.predict_proba(features)[0]
    class_id = int(np.argmax(probs))

    return {
        "class": CLASS_NAMES[class_id],
        "class_id": class_id,
        "confidence": round(float(probs[class_id]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
    }


def save_model(model: XGBClassifier, filepath: str):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {filepath}")


def load_model(filepath: str) -> XGBClassifier:
    """Load trained model from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
