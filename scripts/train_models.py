"""
Model Training Pipeline for ExoplanetAI

End-to-end script:
  1. Generate synthetic data (or load existing)
  2. Preprocess all light curves
  3. Extract features
  4. Train XGBoost with cross-validation
  5. Train 1D CNN
  6. Evaluate and save models
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import time as timer

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_synthetic import generate_dataset, CLASSES
from preprocessing.noise_removal import clean_lightcurve
from preprocessing.detrending import detrend_lightcurve
from detection.bls_detector import run_bls
from features.feature_extractor import extract_features, LightCurveFeatures
from classification.xgboost_classifier import (
    train_xgboost, save_model, CLASS_NAMES, CLASS_MAP
)
from classification.cnn_classifier import (
    train_cnn, save_cnn, prepare_phase_folded_input, PHASE_FOLD_BINS
)


def load_synthetic_data(data_dir: str) -> tuple:
    """Load synthetic dataset from CSV files."""
    labels_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    time_list = []
    flux_list = []
    y_labels = []

    for _, row in labels_df.iterrows():
        fpath = os.path.join(data_dir, row["filename"])
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        time_list.append(df["time"].values)
        flux_list.append(df["flux"].values)
        y_labels.append(CLASS_MAP[row["label"]])

    return time_list, flux_list, np.array(y_labels)


def load_tess_data(data_dir: str) -> tuple:
    """Load real TESS dataset from subdirectories representing classes."""
    time_list = []
    flux_list = []
    y_labels = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"TESS data directory not found: {data_dir}")

    for class_name, label_id in CLASS_MAP.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for fname in os.listdir(class_dir):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(class_dir, fname)
            df = pd.read_csv(fpath)
            time_list.append(df["time"].values)
            flux_list.append(df["flux"].values)
            y_labels.append(label_id)

    if not time_list:
        raise ValueError("No TESS data found. Did you run the fetch_real_data command?")

    return time_list, flux_list, np.array(y_labels)


def process_lightcurves(time_list, flux_list, verbose=True):
    """Clean, detrend, run BLS, and extract features for all light curves."""
    features_list = []
    candidates_list = []
    phase_folded_list = []

    total = len(time_list)
    start = timer.time()

    for i, (t, f) in enumerate(zip(time_list, flux_list)):
        # Clean
        cleaned = clean_lightcurve(t, f)
        flux_clean = cleaned["flux_filtered"]

        # Detrend
        flux_flat = detrend_lightcurve(t, flux_clean, method="polynomial")

        # BLS detection
        candidate = run_bls(t, flux_flat)
        candidates_list.append(candidate)

        # Feature extraction
        feat = extract_features(t, flux_flat, candidate=candidate)
        features_list.append(feat.to_array())

        # Phase-fold for CNN
        if candidate is not None:
            pf = prepare_phase_folded_input(t, flux_flat,
                                             candidate.period, candidate.t0)
        else:
            pf = np.ones(PHASE_FOLD_BINS)
        phase_folded_list.append(pf)

        if verbose and (i + 1) % 50 == 0:
            elapsed = timer.time() - start
            rate = (i + 1) / elapsed
            print(f"  Processed {i+1}/{total} ({rate:.1f} curves/sec)")

    X_features = np.array(features_list)
    X_phase = np.array(phase_folded_list)

    if verbose:
        elapsed = timer.time() - start
        print(f"  ✅ Feature extraction complete: {total} curves in {elapsed:.1f}s")

    return X_features, X_phase, candidates_list


def run_training(n_per_class: int = 400, data_dir: str = None, models_dir: str = None,
                 test_split: float = 0.2, use_real: bool = False):
    """
    Run the full training pipeline.
    
    Parameters
    ----------
    n_per_class : int
        Number of synthetic samples per class to generate (if using synthetic data).
    data_dir : str, optional
        Custom data directory.
    models_dir : str, optional
        Custom models output directory.
    test_split : float
        Fraction of data for validation.
    use_real : bool
        If True, trains on real TESS data in data/tess_labeled instead of synthetic.
    """
    project_root = os.path.join(os.path.dirname(__file__), "..")

    if data_dir is None:
        if use_real:
            data_dir = os.path.join(project_root, "data", "tess_labeled")
        else:
            data_dir = os.path.join(project_root, "data", "synthetic")
            
    if models_dir is None:
        models_dir = os.path.join(project_root, "models")

    os.makedirs(models_dir, exist_ok=True)

    if not use_real:
        # ---- Step 1: Generate synthetic data ----
        print("=" * 60)
        print("STEP 1: Generating synthetic data...")
        print("=" * 60)
        labels_path = os.path.join(data_dir, "labels.csv")
        if not os.path.exists(labels_path):
            generate_dataset(data_dir, n_per_class=n_per_class)
        else:
            print(f"  Using existing dataset in {data_dir}")

        # ---- Step 2: Load data ----
        print("\nSTEP 2: Loading synthetic data...")
        time_list, flux_list, y = load_synthetic_data(data_dir)
    else:
        print("=" * 60)
        print("STEP 1 & 2: Loading real TESS data...")
        print("=" * 60)
        time_list, flux_list, y = load_tess_data(data_dir)

    print(f"  Loaded {len(y)} light curves")
    for cls_name, cls_id in CLASS_MAP.items():
        print(f"    {cls_name}: {(y == cls_id).sum()}")

    # ---- Step 3: Process & extract features ----
    print("\nSTEP 3: Processing light curves & extracting features...")
    X_feat, X_phase, candidates = process_lightcurves(time_list, flux_list)

    # ---- Step 4: Train/val split ----
    print("\nSTEP 4: Splitting data...")
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(y))
    
    stratify_target = y if len(y) > 10 else None
    
    train_idx, val_idx = train_test_split(indices, test_size=test_split,
                                           stratify=stratify_target, random_state=42)

    X_train_feat, X_val_feat = X_feat[train_idx], X_feat[val_idx]
    X_train_phase, X_val_phase = X_phase[train_idx], X_phase[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"  Train: {len(y_train)} | Validation: {len(y_val)}")

    # ---- Step 5: Train XGBoost ----
    print("\n" + "=" * 60)
    print("STEP 5: Training XGBoost classifier...")
    print("=" * 60)
    xgb_result = train_xgboost(X_train_feat, y_train, X_val_feat, y_val)
    print(f"\n  📊 XGBoost Results:")
    print(f"     CV Accuracy: {xgb_result['cv_mean']:.4f} ± {xgb_result['cv_std']:.4f}")
    if "val_accuracy" in xgb_result:
        print(f"     Val Accuracy: {xgb_result['val_accuracy']:.4f}")

    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
    save_model(xgb_result["model"], xgb_path)

    # ---- Step 6: Train CNN ----
    print("\n" + "=" * 60)
    print("STEP 6: Training 1D CNN classifier...")
    print("=" * 60)
    cnn_result = train_cnn(X_train_phase, y_train, X_val_phase, y_val,
                           n_epochs=150, batch_size=32)
    if cnn_result["val_accuracies"]:
        print(f"\n  📊 CNN Final Val Accuracy: {cnn_result['val_accuracies'][-1]:.4f}")

    cnn_path = os.path.join(models_dir, "cnn_model.pt")
    save_cnn(cnn_result["model"], cnn_path)

    # ---- Step 7: Summary ----
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  XGBoost model: {xgb_path}")
    print(f"  CNN model:     {cnn_path}")

    # Save training metrics
    metrics = {
        "xgboost": {
            "cv_mean_accuracy": round(xgb_result["cv_mean"], 4),
            "cv_std": round(xgb_result["cv_std"], 4),
            "val_accuracy": round(xgb_result.get("val_accuracy", 0), 4),
        },
        "cnn": {
            "final_val_accuracy": round(
                cnn_result["val_accuracies"][-1] if cnn_result["val_accuracies"] else 0, 4
            ),
            "final_train_loss": round(cnn_result["train_losses"][-1], 4),
        },
        "dataset": {
            "total_samples": len(y),
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "n_per_class": n_per_class,
        },
    }

    metrics_path = os.path.join(models_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics:       {metrics_path}")

    # Generate evaluation metrics for the dashboard
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
        
        y_pred = xgb_result["model"].predict(X_val_feat)
        y_prob = xgb_result["model"].predict_proba(X_val_feat)
        
        acc = accuracy_score(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted")
        cm = confusion_matrix(y_val, y_pred).tolist()
        
        try:
            roc_auc_list = roc_auc_score(y_val, y_prob, multi_class='ovr', average=None).tolist()
            roc_auc = {name: score for name, score in zip(CLASS_NAMES, roc_auc_list)}
        except:
            roc_auc = {name: 0.0 for name in CLASS_NAMES}
            
        evaluation_metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm,
            "class_labels": CLASS_NAMES,
            "roc_auc_per_class": roc_auc
        }
        eval_path = os.path.join(models_dir, "evaluation_metrics.json")
        with open(eval_path, "w") as f:
            json.dump(evaluation_metrics, f, indent=2)
        print(f"  Eval Metrics:  {eval_path}")
    except Exception as e:
        print(f"  Could not generate evaluation metrics: {e}")

    return metrics


if __name__ == "__main__":
    run_training(n_per_class=50)
