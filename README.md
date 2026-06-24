# 🪐 ExoplanetAI — AI-Enabled Detection of Exoplanets from Noisy Light Curves

An AI-powered astronomical analysis system that automatically detects and classifies exoplanet transit signals from noisy stellar light curve data collected by the **Transiting Exoplanet Survey Satellite (TESS)**.

Built for **ISRO Challenge 7** — combining astronomy, machine learning, signal processing, statistical analysis, and interactive data visualization.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Training Data

```bash
python run.py generate
```

This creates 500 realistic light curves across 5 classes:
- 🌍 **Exoplanet Transits** — Periodic brightness dips from planetary transits
- ⭐ **Eclipsing Binaries** — Deep dips from binary star systems
- 💫 **Variable Stars** — Sinusoidal brightness variations
- 🔀 **Blends** — Signal contamination from nearby sources
- 📡 **Noise** — Instrumental artifacts

### 3. Train ML Models

```bash
python run.py train
```

Trains:
- **XGBoost** classifier on 14 extracted features
- **1D CNN** on phase-folded light curves
- Evaluates with 5-fold cross-validation

### 4. Launch Dashboard

```bash
python run.py dashboard
```

Opens the interactive Streamlit dashboard at `http://localhost:8501`

### 5. Quick Demo (Terminal)

```bash
python run.py demo
```

---

## 📐 Architecture

```
TESS Light Curve Data
        ↓
┌──────────────────────────┐
│   Data Ingestion         │  CSV / FITS upload or TESS download
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   Preprocessing          │  Sigma clipping, median filter, Savitzky-Golay
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   Detrending             │  Polynomial / sliding window normalization
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   BLS Transit Detection  │  Box Least Squares algorithm (astropy)
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   Feature Extraction     │  14 features: depth, period, SNR, symmetry, ...
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   AI Classification      │  XGBoost + 1D CNN Hybrid Ensemble
│   5 Classes              │  Transit | Binary | Variable | Blend | Noise
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   Parameter Estimation   │  Period, depth, Rp/Rs, duration, confidence
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   SHAP Explainability    │  "Why was this classified as a transit?"
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│   Streamlit Dashboard    │  Interactive visualization & reports
│   FastAPI Backend        │  REST API for programmatic access
└──────────────────────────┘
```

---

## 🖥️ Dashboard Features

| Page | Description |
|------|-------------|
| 🏠 Home | Overview stats, classification distribution, recent results |
| 📤 Upload & Analyze | Upload CSV files or load sample data, one-click analysis |
| 📊 Analysis View | Raw/cleaned curves, BLS periodogram, phase-folded transit, parameters |
| 🧠 Explainability | SHAP feature importance, natural language explanations |
| 📦 Batch Processing | Process hundreds of light curves, confusion matrix, CSV export |

---

## 🔌 API Endpoints

Start the API server:
```bash
python run.py api
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload light curve CSV |
| POST | `/analyze` | Run full analysis pipeline |
| POST | `/predict` | Quick classification |
| GET | `/report/{id}` | Get analysis report |
| GET | `/health` | Health check |

---

## 📁 Project Structure

```
ExoplanetAI/
├── data/                    # Light curve data
│   ├── synthetic/           # Generated training data
│   └── tess/                # Real TESS downloads
├── models/                  # Trained ML models
├── preprocessing/           # Noise removal & detrending
├── detection/               # BLS transit detection
├── features/                # Feature extraction
├── classification/          # XGBoost, CNN, hybrid classifier
├── estimation/              # Orbital parameter estimation
├── explainability/          # SHAP explanations
├── backend/                 # FastAPI REST API
├── dashboard/               # Streamlit interactive dashboard
├── scripts/                 # Data generation & training
├── requirements.txt
├── run.py                   # Main CLI entry point
└── README.md
```

---

## 🔬 Tech Stack

| Category | Technology |
|----------|------------|
| **Astronomy** | Astropy, Lightkurve, Astroquery |
| **ML/AI** | PyTorch (1D CNN), XGBoost, Scikit-learn |
| **Signal Processing** | SciPy (Savitzky-Golay, median filter) |
| **Explainability** | SHAP |
| **Backend** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit, Plotly |
| **Data** | NumPy, Pandas |

---

## 👥 Team

Built for ISRO's Challenge 7 hackathon.

---

## 📝 License

MIT License
