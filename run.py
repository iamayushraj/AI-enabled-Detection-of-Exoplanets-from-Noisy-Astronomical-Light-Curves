"""
ExoplanetAI — Main Entry Point

Provides CLI commands to run the full pipeline:
  python run.py generate    — Generate synthetic data
  python run.py train       — Train ML models
  python run.py dashboard   — Launch Streamlit dashboard
  python run.py api         — Launch FastAPI backend
  python run.py demo        — Run a quick demo analysis
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def cmd_generate():
    """Generate synthetic training data."""
    from scripts.generate_synthetic import generate_dataset
    data_dir = os.path.join(PROJECT_ROOT, "data", "synthetic")
    generate_dataset(data_dir, n_per_class=2000, n_points=2000)


def cmd_train(use_real=False):
    """Train ML models."""
    from scripts.train_models import run_training
    run_training(n_per_class=2000, use_real=use_real)

def cmd_fetch_real_data(n_per_class=10):
    """Download real labeled TESS data from NASA Exoplanet Archive."""
    import sys
    from scripts.download_tess import download_labeled_tess
    data_dir = os.path.join(PROJECT_ROOT, "data", "tess_labeled")
    download_labeled_tess(data_dir, n_per_class=n_per_class)


def cmd_dashboard():
    """Launch Streamlit dashboard."""
    dashboard_path = os.path.join(PROJECT_ROOT, "dashboard", "app.py")
    os.system(f"streamlit run \"{dashboard_path}\" --server.port 8501 --theme.base dark")


def cmd_api():
    """Launch FastAPI backend."""
    os.system(f"cd \"{PROJECT_ROOT}\" && uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload")


def cmd_demo():
    """Run a quick demo analysis on synthetic data."""
    import numpy as np
    import pandas as pd

    print("=" * 60)
    print("  ExoplanetAI — Quick Demo")
    print("=" * 60)

    # Generate a single transit light curve
    from scripts.generate_synthetic import generate_transit
    rng = np.random.default_rng(42)
    time = np.linspace(0, 27.4, 2000)
    result = generate_transit(time, rng)

    print(f"\n📡 Generated synthetic transit light curve")
    print(f"   Points: {len(time)}")
    print(f"   True params: {result['params']}")

    # Clean
    from preprocessing.noise_removal import clean_lightcurve
    from preprocessing.detrending import detrend_lightcurve

    cleaned = clean_lightcurve(time, result["flux"])
    flux_flat = detrend_lightcurve(time, cleaned["flux_filtered"])
    print(f"\n🧹 Cleaned: removed {cleaned['n_clipped']} outliers")

    # BLS Detection
    from detection.bls_detector import run_bls
    candidate = run_bls(time, flux_flat)
    if candidate:
        print(f"\n🔍 BLS Detection:")
        print(f"   Period: {candidate.period:.4f} days")
        print(f"   Depth:  {candidate.depth:.6f} ({candidate.depth*100:.4f}%)")
        print(f"   SDE:    {candidate.sde:.2f}")
        print(f"   SNR:    {candidate.snr:.2f}")

    # Feature extraction
    from features.feature_extractor import extract_features
    features = extract_features(time, flux_flat, candidate=candidate)
    print(f"\n🧬 Features extracted: {len(features.to_array())} values")

    # Classification (if models exist)
    from classification.hybrid_classifier import HybridClassifier
    xgb_path = os.path.join(PROJECT_ROOT, "models", "xgboost_model.pkl")
    cnn_path = os.path.join(PROJECT_ROOT, "models", "cnn_model.pt")

    classifier = HybridClassifier(
        xgb_path=xgb_path if os.path.exists(xgb_path) else None,
        cnn_path=cnn_path if os.path.exists(cnn_path) else None,
    )
    prediction = classifier.predict(time, flux_flat)
    print(f"\n🤖 Classification: {prediction['class']}")
    print(f"   Confidence:     {prediction['confidence']:.1f}%")
    print(f"   Probabilities:  {prediction['probabilities']}")

    # Parameter estimation
    if candidate:
        from estimation.parameter_estimator import estimate_parameters
        params = estimate_parameters(time, flux_flat, candidate, prediction)
        print(params.summary_text())

    print("\n✅ Demo complete! Run 'python run.py dashboard' for the full experience.")


def cmd_help():
    print("""
ExoplanetAI — AI-Powered Exoplanet Transit Detection
=====================================================

Usage: python run.py <command>

Commands:
  generate    Generate synthetic training data (500 light curves)
  train       Train XGBoost + CNN classifiers
  dashboard   Launch the Streamlit interactive dashboard
  api         Launch the FastAPI backend server
  demo        Run a quick demo analysis in the terminal

Quick Start:
  1. python run.py generate
  2. python run.py train
  3. python run.py dashboard

Requirements:
  pip install -r requirements.txt
    """)


COMMANDS = {
    "generate": cmd_generate,
    "train": lambda: cmd_train(use_real="--use-real" in sys.argv),
    "fetch_real_data": lambda: cmd_fetch_real_data(int(sys.argv[2]) if len(sys.argv) > 2 else 10),
    "dashboard": cmd_dashboard,
    "api": cmd_api,
    "demo": cmd_demo,
    "help": cmd_help,
    "--help": cmd_help,
    "-h": cmd_help,
}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        cmd_help()
    else:
        cmd = sys.argv[1].lower()
        if cmd in COMMANDS:
            COMMANDS[cmd]()
        else:
            print(f"❌ Unknown command: {cmd}")
            cmd_help()
