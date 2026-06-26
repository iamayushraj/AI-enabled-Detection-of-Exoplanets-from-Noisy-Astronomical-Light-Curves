"""
ExoplanetAI - Interactive Streamlit Dashboard (Hackathon Edition)

Multi-page dashboard for exoplanet transit detection and analysis.

Pages:
  Home              - Animated overview & stats
  Pipeline & Analysis- Upload light curves and run analysis
  Batch Processing   - Process multiple files at once
  Live Fetch (TIC)   - Pull a star directly from NASA/MAST and classify it live
  Explainability     - SHAP feature importance
  Model Metrics      - Confusion matrix, precision/recall, ROC-AUC
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import time as timer
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.noise_removal import clean_lightcurve
from preprocessing.detrending import detrend_lightcurve
from detection.bls_detector import run_bls, phase_fold
from features.feature_extractor import extract_features
from estimation.parameter_estimator import estimate_parameters
from classification.hybrid_classifier import HybridClassifier
from classification.cnn_classifier import prepare_phase_folded_input
from explainability.shap_explainer import explain_prediction, get_global_importance

# Optional: live data fetch. Degrade gracefully if not installed.
try:
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True
except ImportError:
    LIGHTKURVE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ExoplanetAI - Transit Detection Dashboard",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS  (theme + animations)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main, .stApp {
        background: #0A0F1B;
    }

    /* ---------- starfield ---------- */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image:
            radial-gradient(2px 2px at 20px 30px, #ffffff55 0%, transparent 100%),
            radial-gradient(2px 2px at 120px 80px, #ffffff44 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 220px 160px, #ffffff66 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 320px 40px, #ffffff33 0%, transparent 100%),
            radial-gradient(2px 2px at 400px 200px, #ffffff55 0%, transparent 100%),
            radial-gradient(1px 1px at 500px 100px, #ffffff44 0%, transparent 100%);
        background-repeat: repeat;
        background-size: 600px 400px;
        opacity: 0.5;
        z-index: 0;
        pointer-events: none;
        animation: drift 90s linear infinite;
    }
    @keyframes drift {
        from { background-position: 0 0; }
        to   { background-position: -600px 400px; }
    }

    /* ---------- splash / opening animation ---------- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0%, 100% { text-shadow: 0 0 18px rgba(124,131,255,0.55), 0 0 40px rgba(183,148,246,0.25); }
        50%      { text-shadow: 0 0 30px rgba(124,131,255,0.9), 0 0 60px rgba(183,148,246,0.5); }
    }
    @keyframes orbit {
        from { transform: rotate(0deg) translateX(70px) rotate(0deg); }
        to   { transform: rotate(360deg) translateX(70px) rotate(-360deg); }
    }
    .splash-wrap {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        padding: 70px 0 40px 0;
        animation: fadeInUp 1.1s ease-out both;
    }
    .splash-orbit-system {
        position: relative; width: 160px; height: 160px; margin-bottom: 18px;
    }
    .splash-star {
        position: absolute; top: 50%; left: 50%; width: 26px; height: 26px;
        margin: -13px 0 0 -13px; border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #fff7d6, #ffd76b 40%, #ff9f1c 75%);
        box-shadow: 0 0 30px 10px rgba(255, 200, 90, 0.55);
        animation: pulseGlow 2.4s ease-in-out infinite;
    }
    .splash-planet {
        position: absolute; top: 50%; left: 50%; width: 12px; height: 12px;
        margin: -6px 0 0 -6px; border-radius: 50%;
        background: linear-gradient(135deg, #7c83ff, #b794f6);
        animation: orbit 3.2s linear infinite;
        box-shadow: 0 0 12px rgba(124,131,255,0.8);
    }
    .splash-title {
        font-size: 3rem; font-weight: 800; letter-spacing: 1px;
        background: linear-gradient(90deg, #c0c0ff, #b794f6, #7c83ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: pulseGlow 3s ease-in-out infinite;
        margin-bottom: 4px; text-align: center;
    }
    .splash-subtitle {
        color: #9a9ac0; font-size: 1.05rem; letter-spacing: 3px;
        text-transform: uppercase; text-align: center;
    }

    /* ---------- cards ---------- */
    .metric-card {
        background: #121927;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeInUp 0.8s ease-out both;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.15);
        border-color: rgba(139, 92, 246, 0.4);
    }
    .metric-value {
        font-size: 2.5rem; font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.9rem; color: #94A3B8;
        text-transform: uppercase; letter-spacing: 0.5px;
    }

    .result-card {
        background: #121927;
        border: 1px solid #1E293B;
        border-radius: 12px; padding: 24px; margin: 12px 0;
        animation: fadeInUp 0.6s ease-out both;
    }
    .transit-found { background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.4); }
    .no-transit { background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.4); }

    h1, h2, h3 {
        background: linear-gradient(90deg, #c0c0ff, #b794f6, #7c83ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    .stSidebar { background: linear-gradient(180deg, #0d0d2b, #1a1a3e) !important; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #b794f6; }

    .confidence-high { color: #4ade80; font-weight: 700; }
    .confidence-medium { color: #fbbf24; font-weight: 700; }
    .confidence-low { color: #f87171; font-weight: 700; }

    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;
        margin-right: 6px;
    }
    .badge-green { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.4);}
    .badge-amber { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4);}
    .badge-red   { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.4);}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = {}
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False


@st.cache_resource(show_spinner=False)
def load_classifier():
    """Load models once per server process (not per session)."""
    xgb_path = str(PROJECT_ROOT / "models" / "xgboost_model.pkl")
    cnn_path = str(PROJECT_ROOT / "models" / "cnn_model.pt")
    return HybridClassifier(
        xgb_path=xgb_path if os.path.exists(xgb_path) else None,
        cnn_path=cnn_path if os.path.exists(cnn_path) else None,
    )


clf = load_classifier()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sample_data():
    samples = {}
    tess_dir = PROJECT_ROOT / "data" / "tess_labeled"
    if tess_dir.exists():
        for cls_name in ["transit", "eclipsing_binary"]:
            cls_dir = tess_dir / cls_name
            if cls_dir.exists():
                files = list(cls_dir.glob("*.csv"))
                if files:
                    samples[f"Real TESS: {cls_name}"] = files[:5]

    synthetic_dir = PROJECT_ROOT / "data" / "synthetic"
    if synthetic_dir.exists():
        for cls_name in ["transit", "eclipsing_binary", "variable_star", "blend", "noise"]:
            cls_dir = synthetic_dir / cls_name
            if cls_dir.exists():
                files = list(cls_dir.glob("*.csv"))
                if files:
                    samples[f"Synthetic: {cls_name}"] = files[:5]
    return samples


def standardize_columns(df):
    if df is not None and not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns={
            "pdcsap_flux": "flux", "sap_flux": "flux",
            "bjd": "time", "jd": "time",
        })
    return df


def robust_read_csv(file_or_path):
    """Bulletproof CSV loader for messy astronomical files."""
    attempts = [
        dict(on_bad_lines='skip', comment='#'),
        dict(sep=r'\s+', on_bad_lines='skip', comment='#'),
        dict(encoding='latin1', on_bad_lines='skip', comment='#'),
        dict(encoding='latin1', sep=r'\s+', on_bad_lines='skip', comment='#'),
    ]
    last_err = None
    for kwargs in attempts:
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            df = pd.read_csv(file_or_path, **kwargs)
            return standardize_columns(df)
        except Exception as e:
            last_err = e
            continue
    raise last_err


def create_plotly_lightcurve(time, flux, title="Light Curve", color="#7c83ff", show_grid=True):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time, y=flux, mode='markers',
        marker=dict(size=3, color="#8B5CF6", opacity=0.8),
        name="Flux",
        hovertemplate="Time: %{x:.4f} days<br>Flux: %{y:.6f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#F8FAFC")),
        xaxis_title="Time (days)", yaxis_title="Normalized Flux",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8"), height=400,
        margin=dict(l=60, r=30, t=60, b=50), hovermode="x unified",
    )
    if show_grid:
        fig.update_xaxes(gridcolor="rgba(100,100,200,0.15)")
        fig.update_yaxes(gridcolor="rgba(100,100,200,0.15)")
    return fig


def run_full_analysis(time_arr, flux_arr, filename="uploaded", progress_cb=None):
    """Run the complete analysis pipeline. progress_cb(step_fraction, label) optional."""
    steps = [
        ("Cleaning light curve...", lambda: clean_lightcurve(time_arr, flux_arr)),
    ]

    def _tick(frac, label):
        if progress_cb:
            progress_cb(frac, label)

    _tick(0.10, "Cleaning light curve...")
    cleaned = clean_lightcurve(time_arr, flux_arr)
    flux_clean = cleaned["flux_filtered"]

    _tick(0.25, "Detrending...")
    flux_flat = detrend_lightcurve(time_arr, flux_clean)

    _tick(0.45, "Running BLS transit detection...")
    candidate = run_bls(time_arr, flux_flat)

    _tick(0.60, "Extracting features...")
    features = extract_features(time_arr, flux_flat, candidate=candidate)

    _tick(0.75, "Running AI classification...")
    classification = clf.predict(time_arr, flux_flat)

    params = None
    if candidate:
        _tick(0.88, "Estimating parameters...")
        params = estimate_parameters(time_arr, flux_flat, candidate, classification)

    explanation = None
    if clf.xgb_model is not None:
        _tick(0.96, "Generating explanation...")
        try:
            explanation = explain_prediction(clf.xgb_model, features, classification)
        except Exception:
            explanation = None

    _tick(1.0, "Done.")

    return {
        "filename": filename, "time": time_arr, "flux_raw": flux_arr,
        "flux_clean": flux_clean, "flux_flat": flux_flat, "cleaned": cleaned,
        "candidate": candidate, "features": features, "classification": classification,
        "parameters": params, "explanation": explanation,
    }


def animated_counter(placeholder, target, label, suffix=""):
    """Quick count-up animation for a metric card."""
    steps = 12 if target > 0 else 1
    for i in range(steps + 1):
        val = int(target * (i / steps)) if steps else target
        placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}{suffix}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
        if steps:
            timer.sleep(0.025)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🪐 ExoplanetAI")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Home", "Pipeline & Analysis", "Batch Processing", "Live Fetch (TIC)",
         "Explainability (SHAP)", "Model Metrics"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Model Status")
    xgb_status = "🟢 Loaded" if clf.xgb_model is not None else "🔴 Not Found"
    cnn_status = "🟢 Loaded" if clf.cnn_model is not None else "🔴 Not Found"
    st.markdown(f"**XGBoost**: {xgb_status}")
    st.markdown(f"**CNN**: {cnn_status}")

    if clf.xgb_model is None and clf.cnn_model is None:
        st.warning("No models loaded. Run `python run.py train` first.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#606080; font-size:0.8rem;'>"
        "ExoplanetAI v2.0<br>ISRO Challenge 7</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# Shared: show analysis results inline
# =====================================================================
def _show_analysis_results(result):
    st.markdown("---")
    cls = result.get("classification", {})
    params = result.get("parameters")
    predicted_class = cls.get("class", "unknown")
    confidence = cls.get("confidence", 0)

    is_candidate = params and params.is_candidate
    card_class = "transit-found" if is_candidate else "no-transit"
    conf_class = "confidence-high" if confidence > 80 else (
        "confidence-medium" if confidence > 50 else "confidence-low"
    )

    if predicted_class == "transit" and not is_candidate:
        msg = "⚠️ AI predicted Transit, but Physics check failed (radius likely too large, indicating an Eclipsing Binary)."
    elif predicted_class == "transit" and is_candidate:
        msg = "✅ Transit candidate detected and verified by Physics module!"
    else:
        msg = "❌ Not classified as a transit candidate by the AI."

    st.markdown(f"""
    <div class="result-card {card_class}">
        <h2 style="margin:0;">Classification: {predicted_class.replace('_', ' ').title()}</h2>
        <p style="font-size:1.3rem; margin:8px 0;">
            Confidence: <span class="{conf_class}">{confidence:.1f}%</span>
        </p>
        <p>{msg}</p>
    </div>
    """, unsafe_allow_html=True)

    probs = cls.get("probabilities", {})
    if probs:
        prob_df = pd.DataFrame({
            "Class": [k.replace("_", " ").title() for k in probs.keys()],
            "Probability (%)": list(probs.values()),
        })
        fig_prob = px.bar(prob_df, x="Probability (%)", y="Class", orientation="h",
                           color="Probability (%)", color_continuous_scale="Purp")
        fig_prob.update_layout(template="plotly_dark", paper_bgcolor="rgba(10,10,30,0.0)",
                                plot_bgcolor="rgba(10,10,30,0.3)", font=dict(color="#c0c0ff"),
                                height=250, showlegend=False, margin=dict(l=120, r=30, t=30, b=30))
        st.plotly_chart(fig_prob, width='stretch')

    st.markdown("---")
    st.markdown("### Light Curves")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_plotly_lightcurve(result["time"], result["flux_raw"],
                         title="Raw Light Curve", color="#ff6b6b"), width='stretch')
    with col2:
        st.plotly_chart(create_plotly_lightcurve(result["time"], result["flux_flat"],
                         title="Cleaned & Detrended", color="#4ecdc4"), width='stretch')

    candidate = result.get("candidate")
    if candidate and candidate.periods is not None:
        st.markdown("### BLS Periodogram")
        fig_bls = go.Figure()
        fig_bls.add_trace(go.Scattergl(x=candidate.periods, y=candidate.power, mode='lines',
                                        line=dict(color="#b794f6", width=1.5), name="BLS Power"))
        fig_bls.add_vline(x=candidate.period, line_dash="dash", line_color="#4ade80",
                           annotation_text=f"Best Period: {candidate.period:.4f} d",
                           annotation_font_color="#4ade80")
        fig_bls.update_layout(title="Box Least Squares Periodogram", xaxis_title="Period (days)",
                               yaxis_title="BLS Power", template="plotly_dark",
                               paper_bgcolor="rgba(10,10,30,0.0)", plot_bgcolor="rgba(10,10,30,0.3)",
                               font=dict(color="#a0a0c0"), height=350)
        st.plotly_chart(fig_bls, width='stretch')

    if candidate and candidate.period > 0:
        st.markdown("### Phase-Folded Transit")
        phase, flux_folded = phase_fold(result["time"], result["flux_flat"], candidate.period, candidate.t0)
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scattergl(x=phase, y=flux_folded, mode='markers',
                                          marker=dict(color="#7c83ff", size=3, opacity=0.5), name="Data"))
        n_bins = 50
        bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        binned_flux = np.array([
            np.median(flux_folded[(phase >= bin_edges[i]) & (phase < bin_edges[i + 1])])
            if ((phase >= bin_edges[i]) & (phase < bin_edges[i + 1])).sum() > 0 else np.nan
            for i in range(n_bins)
        ])
        fig_phase.add_trace(go.Scatter(x=bin_centres, y=binned_flux, mode='lines+markers',
                                        line=dict(color="#fbbf24", width=2.5), marker=dict(size=5), name="Binned"))
        fig_phase.update_layout(title=f"Phase-Folded on P = {candidate.period:.4f} days",
                                 xaxis_title="Phase", yaxis_title="Normalized Flux",
                                 template="plotly_dark", paper_bgcolor="rgba(10,10,30,0.0)",
                                 plot_bgcolor="rgba(10,10,30,0.3)", font=dict(color="#a0a0c0"), height=400)
        st.plotly_chart(fig_phase, width='stretch')

    if candidate and candidate.period > 0:
        st.markdown("### Detected Transit Events")
        fig_marked = go.Figure()
        fig_marked.add_trace(go.Scatter(x=result["time"], y=result["flux_flat"], mode='lines',
                                         line=dict(color="#4ecdc4", width=1), name="Flux"))
        t0, period, duration = candidate.t0, candidate.period, candidate.duration
        t_min, t_max = result["time"].min(), result["time"].max()
        transit_times, t_current = [], t0
        while t_current < t_max:
            if t_current >= t_min:
                transit_times.append(t_current)
            t_current += period
        for tt in transit_times:
            fig_marked.add_vrect(x0=tt - duration / 2, x1=tt + duration / 2,
                                  fillcolor="rgba(255, 100, 100, 0.15)", line_width=0)
        fig_marked.update_layout(title="Light Curve with Transit Events Marked", xaxis_title="Time (days)",
                                  yaxis_title="Normalized Flux", template="plotly_dark",
                                  paper_bgcolor="rgba(10,10,30,0.0)", plot_bgcolor="rgba(10,10,30,0.3)",
                                  font=dict(color="#a0a0c0"), height=400)
        st.plotly_chart(fig_marked, width='stretch')

    if params:
        st.markdown("### Estimated Parameters")
        param_data = {
            "Parameter": ["Orbital Period", "Transit Depth", "Transit Depth", "Planet/Star Radius Ratio",
                          "Transit Duration", "Impact Parameter", "Number of Transits",
                          "Signal-to-Noise Ratio", "Signal Detection Efficiency", "Detection Confidence"],
            "Value": [f"{params.period_days:.6f} days", f"{params.depth_pct:.4f} %", f"{params.depth_ppm:.0f} ppm",
                      f"{params.radius_ratio:.6f}", f"{params.duration_hours:.2f} hours",
                      f"{params.impact_parameter:.3f}", f"{params.n_transits}", f"{params.snr:.2f}",
                      f"{params.sde:.2f}", f"{params.detection_confidence:.1f} %"],
        }
        st.dataframe(pd.DataFrame(param_data), width='stretch', hide_index=True)

        # Quick export
        export_df = pd.DataFrame(param_data)
        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Parameters (CSV)", csv_bytes,
                            file_name=f"{result['filename']}_parameters.csv", mime="text/csv")

    features = result.get("features")
    if features:
        with st.expander("Extracted Features (click to expand)"):
            st.json(features.to_dict())


# =====================================================================
# PAGE: HOME
# =====================================================================
if page == "Home":

    # ---- opening animation (plays once per session) ----
    if not st.session_state.splash_shown:
        splash = st.empty()
        splash.markdown("""
        <div class="splash-wrap">
            <div class="splash-orbit-system">
                <div class="splash-star"></div>
                <div class="splash-planet"></div>
            </div>
            <div class="splash-title">ExoplanetAI</div>
            <div class="splash-subtitle">Initializing Transit Detection Engine</div>
        </div>
        """, unsafe_allow_html=True)
        timer.sleep(1.3)
        splash.empty()
        st.session_state.splash_shown = True

    st.markdown("# Exoplanet AI Detection Dashboard")
    st.markdown("### AI-Powered Exoplanet Transit Detection from TESS Light Curves")
    st.markdown(
        '<span class="badge badge-green">BLS Detection</span>'
        '<span class="badge badge-green">XGBoost + CNN Ensemble</span>'
        '<span class="badge badge-green">SHAP Explainability</span>'
        '<span class="badge badge-amber">Physics Cross-Check</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    n_analyzed = len(st.session_state.analysis_results)
    n_candidates = sum(
        1 for r in st.session_state.analysis_results.values()
        if r.get("parameters") and r["parameters"].is_candidate
    )
    n_batch = len(st.session_state.batch_results)
    models_count = sum([clf.xgb_model is not None, clf.cnn_model is not None])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ph1 = st.empty()
    with col2:
        ph2 = st.empty()
    with col3:
        ph3 = st.empty()
    with col4:
        ph4 = st.empty()

    animated_counter(ph1, n_analyzed, "Stars Analyzed")
    animated_counter(ph2, n_candidates, "Transit Candidates")
    animated_counter(ph3, n_batch, "Batch Results")
    ph4.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{models_count}/2</div>
        <div class="metric-label">Models Loaded</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Analysis Pipeline")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Data Ingestion** - Upload CSV/FITS light curve data, or fetch live from MAST by TIC ID

        **Preprocessing** - Sigma clipping, median filtering, Savitzky-Golay smoothing

        **Detrending** - Remove long-term stellar variability

        **Transit Detection** - Box Least Squares (BLS) algorithm

        **Feature Extraction** - 14 ML features (depth, period, SNR, symmetry, etc.)
        """)
    with col_b:
        st.markdown("""
        **AI Classification** - XGBoost + 1D CNN hybrid ensemble

        **5 Classes**: Transit, Eclipsing Binary, Variable Star, Blend, Noise

        **Parameter Estimation** - Period, depth, radius ratio, impact parameter

        **Explainability** - SHAP feature importance analysis

        **Confidence Scoring** - Composite detection confidence, cross-checked by physics
        """)

    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("### Classification Distribution")
        classes = [r["classification"]["class"] for r in st.session_state.analysis_results.values()
                   if r.get("classification")]
        if classes:
            class_counts = pd.Series(classes).value_counts()
            fig = px.pie(values=class_counts.values, names=class_counts.index,
                         color_discrete_sequence=px.colors.sequential.Purp, hole=0.4)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(10,10,30,0.0)",
                              font=dict(color="#c0c0ff"), height=350)
            st.plotly_chart(fig, width='stretch')

    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("### Recent Analyses")
        rows = []
        for name, r in list(st.session_state.analysis_results.items())[-10:]:
            cls = r.get("classification", {})
            params = r.get("parameters")
            rows.append({
                "File": name, "Class": cls.get("class", "-"),
                "Confidence": f"{cls.get('confidence', 0):.1f}%",
                "Period (days)": f"{params.period_days:.4f}" if params else "-",
                "Candidate": "YES" if (params and params.is_candidate) else "NO",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# =====================================================================
# PAGE: PIPELINE & ANALYSIS
# =====================================================================
elif page == "Pipeline & Analysis":
    st.markdown("# Upload & Analyze Light Curves")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Upload File", "Load Sample Data"])

    with tab1:
        uploaded_file = st.file_uploader("Upload a light curve CSV file", type=["csv"],
                                          help="CSV file with 'time' and 'flux' columns")
        if uploaded_file:
            try:
                df = robust_read_csv(uploaded_file)
            except Exception as e:
                df = None
                st.error(f"Could not parse file: {e}")

            if df is None or "time" not in df.columns or "flux" not in df.columns:
                st.error("CSV must have 'time' and 'flux' columns!")
            else:
                st.success(f"Loaded **{uploaded_file.name}** - {len(df)} data points")
                st.plotly_chart(create_plotly_lightcurve(df["time"].values, df["flux"].values,
                                 title=f"Raw Light Curve: {uploaded_file.name}"), width='stretch')

                if st.button("Run Full Analysis", type="primary", key="analyze_upload"):
                    progress = st.progress(0, text="Starting...")

                    def cb(frac, label):
                        progress.progress(frac, text=label)

                    result = run_full_analysis(df["time"].values, df["flux"].values,
                                                filename=uploaded_file.name, progress_cb=cb)
                    progress.empty()
                    st.session_state.analysis_results[uploaded_file.name] = result
                    st.session_state.uploaded_data[uploaded_file.name] = df
                    st.success("Analysis complete! See results below.")
                    _show_analysis_results(result)

    with tab2:
        samples = load_sample_data()
        if not samples:
            st.warning("No synthetic data found. Run `python run.py generate` first.")
        else:
            st.markdown("### Select a sample light curve:")
            sample_class = st.selectbox("Signal Class", list(samples.keys()),
                format_func=lambda x: {"transit": "Exoplanet Transit", "eclipsing_binary": "Eclipsing Binary",
                                        "variable_star": "Variable Star", "blend": "Blend",
                                        "noise": "Noise"}.get(x, x))
            sample_files = samples[sample_class]
            sample_file = st.selectbox("Select file", sample_files, format_func=lambda x: x.name)

            if sample_file:
                try:
                    df = robust_read_csv(sample_file)
                except Exception as e:
                    df = None
                    st.error(f"Could not parse sample: {e}")

                if df is not None:
                    st.info(f"{sample_file.name} - {len(df)} points - Class: **{sample_class}**")
                    st.plotly_chart(create_plotly_lightcurve(df["time"].values, df["flux"].values,
                                     title=f"Sample: {sample_file.name} ({sample_class})"), width='stretch')

                    if st.button("Run Full Analysis", type="primary", key="analyze_sample"):
                        progress = st.progress(0, text="Starting...")

                        def cb(frac, label):
                            progress.progress(frac, text=label)

                        result = run_full_analysis(df["time"].values, df["flux"].values,
                                                    filename=sample_file.name, progress_cb=cb)
                        progress.empty()
                        st.session_state.analysis_results[sample_file.name] = result
                        st.success("Analysis complete! See results below.")
                        _show_analysis_results(result)


# =====================================================================
# PAGE: BATCH PROCESSING
# =====================================================================
elif page == "Batch Processing":
    st.markdown("# Batch Processing")
    st.markdown("### Classify multiple light curves in one pass")
    st.markdown("---")

    batch_files = st.file_uploader("Upload multiple light curve CSVs", type=["csv"],
                                    accept_multiple_files=True)

    if batch_files:
        st.info(f"{len(batch_files)} file(s) ready to process.")

        if st.button("Run Batch Analysis", type="primary"):
            progress = st.progress(0, text="Starting batch...")
            results_table = []
            errors = []

            for i, f in enumerate(batch_files):
                try:
                    df = robust_read_csv(f)
                    if df is None or "time" not in df.columns or "flux" not in df.columns:
                        errors.append((f.name, "missing time/flux columns"))
                        continue

                    result = run_full_analysis(df["time"].values, df["flux"].values, filename=f.name)
                    st.session_state.analysis_results[f.name] = result

                    cls = result.get("classification", {})
                    params = result.get("parameters")
                    results_table.append({
                        "File": f.name,
                        "Class": cls.get("class", "-"),
                        "Confidence (%)": round(cls.get("confidence", 0), 1),
                        "Period (days)": round(params.period_days, 4) if params else None,
                        "Depth (ppm)": round(params.depth_ppm, 0) if params else None,
                        "SNR": round(params.snr, 2) if params else None,
                        "Candidate": "YES" if (params and params.is_candidate) else "NO",
                    })
                except Exception as e:
                    errors.append((f.name, str(e)))

                progress.progress((i + 1) / len(batch_files), text=f"Processed {f.name}")

            progress.empty()
            st.session_state.batch_results = results_table

            if results_table:
                st.success(f"Batch complete: {len(results_table)} processed, {len(errors)} skipped.")
            if errors:
                with st.expander(f"⚠️ {len(errors)} file(s) had issues"):
                    for name, err in errors:
                        st.write(f"**{name}**: {err}")

    if st.session_state.batch_results:
        st.markdown("---")
        st.markdown("### Batch Results")
        batch_df = pd.DataFrame(st.session_state.batch_results)
        st.dataframe(batch_df, width='stretch', hide_index=True)

        n_candidates = (batch_df["Candidate"] == "YES").sum() if "Candidate" in batch_df else 0
        st.markdown(f"**{n_candidates} / {len(batch_df)}** flagged as transit candidates.")

        # Class distribution
        if "Class" in batch_df.columns:
            fig = px.pie(batch_df, names="Class", color_discrete_sequence=px.colors.sequential.Purp, hole=0.4)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(10,10,30,0.0)",
                              font=dict(color="#c0c0ff"), height=350)
            st.plotly_chart(fig, width='stretch')

        csv_bytes = batch_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Batch Results (CSV)", csv_bytes,
                            file_name="batch_results.csv", mime="text/csv")
    else:
        st.info("Upload files above and click **Run Batch Analysis** to populate this page.")


# =====================================================================
# PAGE: LIVE FETCH (TIC)
# =====================================================================
elif page == "Live Fetch (TIC)":
    st.markdown("# Live Fetch from NASA / MAST")
    st.markdown("### Pull a real TESS light curve by TIC ID and classify it live")
    st.markdown("---")

    if not LIGHTKURVE_AVAILABLE:
        st.warning(
            "`lightkurve` is not installed in this environment, so live MAST fetching is disabled.\n\n"
            "To enable it: add `lightkurve` to `requirements.txt` and redeploy. "
            "Once installed, this page will query MAST directly by TIC ID and run the same "
            "pipeline used elsewhere in the dashboard."
        )
        st.code("pip install lightkurve", language="bash")
    else:
        tic_id = st.text_input("Enter TIC ID", placeholder="e.g. 307210830 (TOI-700)")
        col1, col2 = st.columns(2)
        with col1:
            mission = st.selectbox("Mission", ["TESS"], index=0)
        with col2:
            author = st.selectbox("Pipeline", ["SPOC", "QLP"], index=0)

        if st.button("Fetch & Analyze", type="primary") and tic_id:
            with st.spinner(f"Searching MAST for TIC {tic_id}..."):
                try:
                    search = lk.search_lightcurve(f"TIC {tic_id}", mission=mission, author=author)
                    if len(search) == 0:
                        st.error("No light curves found for that TIC ID / pipeline combination.")
                    else:
                        lc = search[0].download()
                        lc = lc.remove_nans()
                        time_arr = np.asarray(lc.time.value, dtype=np.float64)
                        flux_arr = np.asarray(lc.flux.value, dtype=np.float64)
                        st.success(f"Fetched {len(time_arr)} data points for TIC {tic_id}.")

                        st.plotly_chart(create_plotly_lightcurve(time_arr, flux_arr,
                                         title=f"TIC {tic_id} - Raw Light Curve"), width='stretch')

                        progress = st.progress(0, text="Starting...")

                        def cb(frac, label):
                            progress.progress(frac, text=label)

                        result = run_full_analysis(time_arr, flux_arr, filename=f"TIC_{tic_id}", progress_cb=cb)
                        progress.empty()
                        st.session_state.analysis_results[f"TIC_{tic_id}"] = result
                        _show_analysis_results(result)
                except Exception as e:
                    st.error(f"Fetch failed: {e}")


# =====================================================================
# PAGE: EXPLAINABILITY
# =====================================================================
elif page == "Explainability (SHAP)":
    st.markdown("# AI Explainability")
    st.markdown("### Why did the model make this prediction?")
    st.markdown("---")

    if not st.session_state.analysis_results:
        st.info("No analyses yet. Run an analysis first from the Pipeline & Analysis page.")
    else:
        analysis_names = list(st.session_state.analysis_results.keys())
        selected = st.selectbox("Select analysis:", analysis_names, index=len(analysis_names) - 1)
        result = st.session_state.analysis_results[selected]
        explanation = result.get("explanation")

        if explanation is None:
            st.warning("No explainability data. Models may not be loaded.")
        else:
            predicted_class = result["classification"]["class"]
            is_candidate = result.get("parameters") and result["parameters"].is_candidate

            st.markdown("### Explanation")
            if predicted_class == "transit" and not is_candidate:
                st.warning("⚠️ **Note:** The AI predicted Transit, but the Physics module rejected it "
                           "(implied planetary radius is unphysical). The SHAP chart below explains why "
                           "the AI leaned toward Transit anyway.")

            st.markdown(f"""
            <div class="result-card">
                <pre style="color: #c0c0ff; white-space: pre-wrap; font-family: Inter, sans-serif;">
{explanation.get('explanation_text', 'No explanation available.')}
                </pre>
                <p style="color: #606080; font-size: 0.8rem;">Method: {explanation.get('method', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

            contributions = explanation.get("contributions", [])
            if contributions:
                st.markdown("### Feature Contributions")
                feat_names = [c["feature"].replace("_", " ").title() for c in contributions[:10]]
                feat_importance = [c.get("importance_pct", abs(c.get("shap_value", 0)) * 100) for c in contributions[:10]]
                feat_colors = ["#4ade80" if c.get("impact") == "positive" or c.get("shap_value", 0) > 0
                               else "#f87171" for c in contributions[:10]]

                fig_shap = go.Figure()
                fig_shap.add_trace(go.Bar(y=feat_names[::-1], x=feat_importance[::-1], orientation='h',
                                           marker_color=feat_colors[::-1]))
                fig_shap.update_layout(title="Feature Importance (Top 10)", xaxis_title="Importance (%)",
                                       template="plotly_dark", paper_bgcolor="rgba(10,10,30,0.0)",
                                       plot_bgcolor="rgba(10,10,30,0.3)", font=dict(color="#c0c0ff"),
                                       height=400, margin=dict(l=160, r=30, t=50, b=40))
                st.plotly_chart(fig_shap, width='stretch')

            if clf.xgb_model is not None:
                st.markdown("### Global Feature Importance")
                global_imp = get_global_importance(clf.xgb_model)
                if global_imp:
                    g_names = list(global_imp.keys())[:10]
                    g_vals = list(global_imp.values())[:10]
                    fig_global = px.bar(x=g_vals[::-1], y=[n.replace("_", " ").title() for n in g_names[::-1]],
                                        orientation='h', color=g_vals[::-1], color_continuous_scale="Purp")
                    fig_global.update_layout(template="plotly_dark", paper_bgcolor="rgba(10,10,30,0.0)",
                                             plot_bgcolor="rgba(10,10,30,0.3)", font=dict(color="#c0c0ff"),
                                             height=380, showlegend=False, margin=dict(l=160, r=30, t=30, b=40),
                                             xaxis_title="Importance")
                    st.plotly_chart(fig_global, width='stretch')


# =====================================================================
# PAGE: MODEL METRICS
# =====================================================================
elif page == "Model Metrics":
    st.markdown("# Model Performance")
    st.markdown("### How good is the classifier, really?")
    st.markdown("---")

    metrics_path = PROJECT_ROOT / "models" / "evaluation_metrics.json"

    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        for col, key, label in zip(
            [col1, col2, col3, col4],
            ["accuracy", "precision", "recall", "f1"],
            ["Accuracy", "Precision", "Recall", "F1 Score"],
        ):
            val = metrics.get(key)
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{f'{val:.1%}' if val is not None else '—'}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        cm = metrics.get("confusion_matrix")
        labels = metrics.get("class_labels", ["transit", "eclipsing_binary", "variable_star", "blend", "noise"])
        if cm:
            st.markdown("---")
            st.markdown("### Confusion Matrix")
            fig_cm = px.imshow(cm, x=labels, y=labels, color_continuous_scale="Purp",
                                labels=dict(x="Predicted", y="Actual", color="Count"), text_auto=True)
            fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(10,10,30,0.0)",
                                 font=dict(color="#c0c0ff"), height=450)
            st.plotly_chart(fig_cm, width='stretch')

        roc = metrics.get("roc_auc_per_class")
        if roc:
            st.markdown("### ROC-AUC per Class")
            roc_df = pd.DataFrame({"Class": list(roc.keys()), "AUC": list(roc.values())})
            fig_roc = px.bar(roc_df, x="Class", y="AUC", color="AUC", color_continuous_scale="Purp", range_y=[0, 1])
            fig_roc.update_layout(template="plotly_dark", paper_bgcolor="rgba(10,10,30,0.0)",
                                  plot_bgcolor="rgba(10,10,30,0.3)", font=dict(color="#c0c0ff"), height=350)
            st.plotly_chart(fig_roc, width='stretch')
    else:
        st.info(
            "No `models/evaluation_metrics.json` found yet. Generate one from your training script "
            "(e.g. via `sklearn.metrics.classification_report` + `confusion_matrix`) and save as JSON "
            "with keys: `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `class_labels`, "
            "`roc_auc_per_class`. This page will pick it up automatically."
        )
        st.code('''
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred).tolist()

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision, "recall": recall, "f1": f1,
    "confusion_matrix": cm,
    "class_labels": ["transit", "eclipsing_binary", "variable_star", "blend", "noise"],
}
with open("models/evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
''', language="python")
