"""
Pydantic Schemas for ExoplanetAI FastAPI Backend
"""

from pydantic import BaseModel, Field
from typing import Optional


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    n_points: int
    message: str


class AnalysisRequest(BaseModel):
    file_id: str
    run_bls: bool = True
    run_classification: bool = True
    run_estimation: bool = True
    run_explainability: bool = True


class TransitCandidateSchema(BaseModel):
    period: float = 0
    depth: float = 0
    depth_pct: float = 0
    duration_hours: float = 0
    t0: float = 0
    bls_power: float = 0
    sde: float = 0
    snr: float = 0
    n_transits: int = 0


class ClassificationSchema(BaseModel):
    predicted_class: str = "unknown"
    confidence: float = 0
    probabilities: dict = {}


class ParameterSchema(BaseModel):
    period_days: float = 0
    period_uncertainty_days: float = 0
    depth_pct: float = 0
    depth_ppm: float = 0
    radius_ratio_Rp_Rs: float = 0
    duration_hours: float = 0
    impact_parameter: float = 0
    n_transits: int = 0
    snr: float = 0
    sde: float = 0
    detection_confidence_pct: float = 0
    is_candidate: bool = False


class ExplainabilitySchema(BaseModel):
    explanation_text: str = ""
    method: str = ""
    contributions: list = []


class FullAnalysisResponse(BaseModel):
    file_id: str
    filename: str
    status: str = "success"
    transit_candidate: Optional[TransitCandidateSchema] = None
    classification: Optional[ClassificationSchema] = None
    parameters: Optional[ParameterSchema] = None
    explainability: Optional[ExplainabilitySchema] = None
    processing_time_seconds: float = 0


class BatchStatusResponse(BaseModel):
    total: int = 0
    processed: int = 0
    transit_candidates: int = 0
    status: str = "idle"


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
