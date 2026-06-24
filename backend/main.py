"""
FastAPI Backend for ExoplanetAI

Endpoints:
  POST /upload      — Upload light curve file
  POST /analyze     — Run full analysis pipeline
  POST /predict     — Quick classification only
  GET  /report/{id} — Get analysis report
  GET  /health      — Health check
"""

import os
import sys
import uuid
import time as timer
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.schemas import (
    UploadResponse, FullAnalysisResponse, TransitCandidateSchema,
    ClassificationSchema, ParameterSchema, ExplainabilitySchema,
    BatchStatusResponse, ErrorResponse
)
from preprocessing.noise_removal import clean_lightcurve
from preprocessing.detrending import detrend_lightcurve
from detection.bls_detector import run_bls
from features.feature_extractor import extract_features
from estimation.parameter_estimator import estimate_parameters
from classification.hybrid_classifier import HybridClassifier
from explainability.shap_explainer import explain_prediction

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# File store  { file_id: { path, filename, n_points } }
file_store: dict = {}

# Analysis results store { file_id: result_dict }
results_store: dict = {}

# Models
classifier: Optional[HybridClassifier] = None

def _load_models():
    """Load ML models on startup."""
    global classifier
    xgb_path = str(PROJECT_ROOT / "models" / "xgboost_model.pkl")
    cnn_path = str(PROJECT_ROOT / "models" / "cnn_model.pt")
    classifier = HybridClassifier(
        xgb_path=xgb_path if os.path.exists(xgb_path) else None,
        cnn_path=cnn_path if os.path.exists(cnn_path) else None,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ExoplanetAI API",
    description="AI-powered exoplanet transit detection from TESS light curves",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": classifier is not None,
        "xgb_loaded": classifier.xgb_model is not None if classifier else False,
        "cnn_loaded": classifier.cnn_model is not None if classifier else False,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a light curve CSV file."""
    try:
        content = await file.read()
        file_id = str(uuid.uuid4())[:8]

        # Save file
        filepath = UPLOAD_DIR / f"{file_id}_{file.filename}"
        with open(filepath, "wb") as f:
            f.write(content)

        # Parse to verify
        df = pd.read_csv(filepath)
        if "time" not in df.columns or "flux" not in df.columns:
            os.remove(filepath)
            raise HTTPException(400, "CSV must have 'time' and 'flux' columns")

        file_store[file_id] = {
            "path": str(filepath),
            "filename": file.filename,
            "n_points": len(df),
        }

        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            n_points=len(df),
            message=f"File uploaded successfully ({len(df)} data points)",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


@app.post("/analyze", response_model=FullAnalysisResponse)
async def analyze(file_id: str):
    """Run full analysis pipeline on an uploaded file."""
    if file_id not in file_store:
        raise HTTPException(404, f"File ID '{file_id}' not found")

    start_time = timer.time()
    info = file_store[file_id]

    try:
        # Load data
        df = pd.read_csv(info["path"])
        time_arr = df["time"].values.astype(float)
        flux_arr = df["flux"].values.astype(float)

        # Step 1: Clean
        cleaned = clean_lightcurve(time_arr, flux_arr)
        flux_clean = cleaned["flux_filtered"]

        # Step 2: Detrend
        flux_flat = detrend_lightcurve(time_arr, flux_clean)

        # Step 3: BLS detection
        candidate = run_bls(time_arr, flux_flat)

        # Step 4: Feature extraction
        features = extract_features(time_arr, flux_flat, candidate=candidate)

        # Step 5: Classification
        classification_result = None
        if classifier is not None:
            classification_result = classifier.predict(time_arr, flux_flat)

        # Step 6: Parameter estimation
        params_result = None
        if candidate is not None:
            params = estimate_parameters(time_arr, flux_flat, candidate,
                                          classification_result)
            params_result = params.to_dict()

        # Step 7: Explainability
        explain_result = None
        if classifier is not None and classifier.xgb_model is not None and classification_result:
            explain_result = explain_prediction(
                classifier.xgb_model, features, classification_result
            )

        elapsed = timer.time() - start_time

        # Build response
        response = FullAnalysisResponse(
            file_id=file_id,
            filename=info["filename"],
            transit_candidate=TransitCandidateSchema(
                **(candidate.to_dict() if candidate else {})
            ),
            classification=ClassificationSchema(
                predicted_class=classification_result.get("class", "unknown") if classification_result else "unknown",
                confidence=classification_result.get("confidence", 0) if classification_result else 0,
                probabilities=classification_result.get("probabilities", {}) if classification_result else {},
            ),
            parameters=ParameterSchema(**(params_result or {})),
            explainability=ExplainabilitySchema(
                explanation_text=explain_result.get("explanation_text", "") if explain_result else "",
                method=explain_result.get("method", "") if explain_result else "",
                contributions=explain_result.get("contributions", []) if explain_result else [],
            ),
            processing_time_seconds=round(elapsed, 3),
        )

        results_store[file_id] = response.model_dump()
        return response

    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.post("/predict")
async def quick_predict(file_id: str):
    """Quick classification only (no parameter estimation)."""
    if file_id not in file_store:
        raise HTTPException(404, f"File ID '{file_id}' not found")

    info = file_store[file_id]
    df = pd.read_csv(info["path"])
    time_arr = df["time"].values.astype(float)
    flux_arr = df["flux"].values.astype(float)

    cleaned = clean_lightcurve(time_arr, flux_arr)
    flux_flat = detrend_lightcurve(time_arr, cleaned["flux_filtered"])

    if classifier is not None:
        result = classifier.predict(time_arr, flux_flat)
        return result
    else:
        raise HTTPException(503, "No models loaded")


@app.get("/report/{file_id}")
async def get_report(file_id: str):
    """Get stored analysis report."""
    if file_id not in results_store:
        raise HTTPException(404, f"No analysis found for file ID '{file_id}'")
    return results_store[file_id]


@app.get("/files")
async def list_files():
    """List all uploaded files."""
    return {
        fid: {"filename": info["filename"], "n_points": info["n_points"]}
        for fid, info in file_store.items()
    }


# ---------------------------------------------------------------------------
# Run with: uvicorn backend.main:app --reload --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
