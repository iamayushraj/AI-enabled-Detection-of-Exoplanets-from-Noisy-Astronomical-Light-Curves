# 🪐 ExoplanetAI — AI-Enabled Detection of Exoplanets from Noisy Light Curves

An advanced, AI-powered astronomical analysis system that automatically detects and classifies exoplanet transit signals from noisy stellar light curve data collected by the **Transiting Exoplanet Survey Satellite (TESS)**.

Built for **ISRO Challenge 7** — combining astronomy, machine learning, signal processing, statistical analysis, and interactive data visualization into a stunning Dark Neon dashboard.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Training Data

```bash
python scripts/generate_synthetic.py
```

This creates realistic light curves across 5 classes:
- 🌍 **Exoplanet Transits** — Periodic brightness dips from planetary transits
- ⭐ **Eclipsing Binaries** — Deep dips from binary star systems
- 💫 **Variable Stars** — Sinusoidal brightness variations
- 🔀 **Blends** — Signal contamination from nearby sources
- 📡 **Noise** — Instrumental artifacts

### 3. Train ML Models

```bash
python scripts/train_models.py
```

Trains the dual-engine AI:
- **XGBoost** classifier on 11 extracted physical features
- **1D CNN** on phase-folded light curves
- Generates `models/evaluation_metrics.json` automatically

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Opens the interactive Streamlit dashboard at `http://localhost:8501`.

---

## 📐 Architecture

```text
TESS Light Curve Data
        ↓
┌──────────────────────────┐
│   Data Ingestion         │  CSV Upload or Live MAST Fetch via TIC ID
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   Preprocessing          │  Sigma clipping, float64 casting, interpolation
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   Detrending             │  Polynomial smoothing and flattening
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   BLS Transit Detection  │  Box Least Squares algorithm (astropy)
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   Feature Extraction     │  11 features: depth, period, SNR, symmetry, ...
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   AI Classification      │  XGBoost + 1D CNN Hybrid Ensemble
│   5 Classes              │  Transit | Binary | Variable | Blend | Noise
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   Parameter Estimation   │  Period, depth, duration, confidence scoring
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   SHAP Explainability    │  Interpretable AI feature importance
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   Streamlit Dashboard    │  Interactive visualizations & Dark Neon UI
└──────────────────────────┘
```

---

## 🖥️ Dashboard Features

| Page | Description |
|------|-------------|
| 🏠 **Home** | Overview stats, animated system visualization, loaded models status |
| 📤 **Pipeline & Analysis** | Upload local CSV files, detrending, detection, and parameter estimation |
| 📦 **Batch Processing** | Process multiple light curves in one pass with aggregate statistics |
| 📡 **Live Fetch (TIC)** | Directly pull TESS light curves from NASA's MAST archive via TIC ID |
| 🧠 **Explainability (SHAP)** | Deep learning model explainability with feature importance breakdowns |
| 📈 **Model Metrics** | Confusion matrix, ROC-AUC curves, and real-time validation accuracy (94.5%) |

---

## 📁 Project Structure

```text
ExoplanetAI/
├── classification/          # XGBoost, CNN, hybrid classifier
├── dashboard/               # Streamlit interactive dashboard (app.py)
├── data/                    # Light curve data (synthetic/tess)
├── detection/               # BLS transit detection
├── estimation/              # Orbital parameter estimation
├── explainability/          # SHAP explanations (shap_explainer.py)
├── features/                # Feature extraction pipeline
├── models/                  # Trained ML models and evaluation metrics
├── preprocessing/           # Noise removal & detrending (float64 compliant)
├── scripts/                 # Data generation & training
├── requirements.txt         # Dependency tree
└── README.md                # Project documentation
```

---

## 🔬 Tech Stack

| Category | Technology |
|----------|------------|
| **Astronomy** | Astropy, Lightkurve (NASA MAST APIs) |
| **ML/AI** | PyTorch (1D CNN), XGBoost, Scikit-learn |
| **Signal Processing** | SciPy, Numpy |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Dashboard** | Streamlit, Plotly Express, Plotly Graph Objects |

---

## 👥 Team

Engineered and designed for **ISRO's Challenge 7** Hackathon. 

---

## 📝 License

MIT License
