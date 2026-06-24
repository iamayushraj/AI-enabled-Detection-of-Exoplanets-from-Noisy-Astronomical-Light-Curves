"""
ExoplanetAI - Interactive Streamlit Dashboard

Multi-page dashboard for exoplanet transit detection and analysis.

Pages:
  Home           - Overview & stats
  Upload & Analyze - Upload light curves and run analysis
  Analysis View   - Detailed plots and visualizations
  Explainability  - SHAP feature importance
  Batch Processing - Process multiple files
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import json
import time as timer
from pathlib import Path
from io import StringIO

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


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ExoplanetAI - Transit Detection Dashboard",
    page_icon="icon",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d0d2b 100%); }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d0d2b 100%);
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(30, 30, 80, 0.8), rgba(50, 50, 120, 0.6));
        border: 1px solid rgba(100, 100, 255, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(80, 80, 255, 0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7c83ff, #b794f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0c0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(20, 60, 40, 0.8), rgba(30, 80, 50, 0.6));
        border: 1px solid rgba(100, 255, 150, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
    }

    .transit-found {
        background: linear-gradient(135deg, rgba(20, 80, 40, 0.9), rgba(30, 120, 60, 0.7));
        border: 1px solid rgba(80, 255, 120, 0.4);
    }

    .no-transit {
        background: linear-gradient(135deg, rgba(80, 30, 30, 0.8), rgba(120, 40, 40, 0.6));
        border: 1px solid rgba(255, 100, 100, 0.3);
    }

    h1, h2, h3 {
        background: linear-gradient(90deg, #c0c0ff, #b794f6, #7c83ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stSidebar {
        background: linear-gradient(180deg, #0d0d2b, #1a1a3e) !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #b794f6;
    }

    .confidence-high { color: #4ade80; font-weight: 700; }
    .confidence-medium { color: #fbbf24; font-weight: 700; }
    .confidence-low { color: #f87171; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = {}
if "classifier" not in st.session_state:
    # Load models once
    xgb_path = str(PROJECT_ROOT / "models" / "xgboost_model.pkl")
    cnn_path = str(PROJECT_ROOT / "models" / "cnn_model.pt")
    st.session_state.classifier = HybridClassifier(
        xgb_path=xgb_path if os.path.exists(xgb_path) else None,
        cnn_path=cnn_path if os.path.exists(cnn_path) else None,
    )
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sample_data():
    """Load sample data for demo."""
    samples = {}
    
    # Try loading from real TESS data first
    tess_dir = PROJECT_ROOT / "data" / "tess_labeled"
    if tess_dir.exists():
        for cls_name in ["transit", "eclipsing_binary"]:
            cls_dir = tess_dir / cls_name
            if cls_dir.exists():
                files = list(cls_dir.glob("*.csv"))
                if files:
                    samples[f"Real TESS: {cls_name}"] = files[:5]

    # Add synthetic data as well
    synthetic_dir = PROJECT_ROOT / "data" / "synthetic"
    if synthetic_dir.exists():
        for cls_name in ["transit", "eclipsing_binary", "variable_star", "blend", "noise"]:
            cls_dir = synthetic_dir / cls_name
            if cls_dir.exists():
                files = list(cls_dir.glob("*.csv"))
                if files:
                    samples[f"Synthetic: {cls_name}"] = files[:5]

    return samples


def create_plotly_lightcurve(time, flux, title="Light Curve", color="#7c83ff",
                              show_grid=True):
    """Create a styled Plotly light curve figure."""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time, y=flux, mode='lines',
        line=dict(color=color, width=1.5),
        name="Flux",
        hovertemplate="Time: %{x:.4f} days<br>Flux: %{y:.6f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#c0c0ff")),
        xaxis_title="Time (days)",
        yaxis_title="Normalized Flux",
        template="plotly_dark",
        paper_bgcolor="rgba(10,10,30,0.8)",
        plot_bgcolor="rgba(10,10,30,0.6)",
        font=dict(color="#a0a0c0"),
        height=400,
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
    )
    if show_grid:
        fig.update_xaxes(gridcolor="rgba(100,100,200,0.15)")
        fig.update_yaxes(gridcolor="rgba(100,100,200,0.15)")
    return fig


def run_full_analysis(time_arr, flux_arr, filename="uploaded"):
    """Run the complete analysis pipeline."""
    with st.spinner("Cleaning light curve..."):
        cleaned = clean_lightcurve(time_arr, flux_arr)
        flux_clean = cleaned["flux_filtered"]

    with st.spinner("Detrending..."):
        flux_flat = detrend_lightcurve(time_arr, flux_clean)

    with st.spinner("Running BLS transit detection..."):
        candidate = run_bls(time_arr, flux_flat)

    with st.spinner("Extracting features..."):
        features = extract_features(time_arr, flux_flat, candidate=candidate)

    with st.spinner("Running AI classification..."):
        clf = st.session_state.classifier
        classification = clf.predict(time_arr, flux_flat)

    params = None
    if candidate:
        with st.spinner("Estimating parameters..."):
            params = estimate_parameters(time_arr, flux_flat, candidate, classification)

    explanation = None
    if clf.xgb_model is not None:
        with st.spinner("Generating explanation..."):
            try:
                explanation = explain_prediction(clf.xgb_model, features, classification)
            except Exception:
                explanation = None

    return {
        "filename": filename,
        "time": time_arr,
        "flux_raw": flux_arr,
        "flux_clean": flux_clean,
        "flux_flat": flux_flat,
        "cleaned": cleaned,
        "candidate": candidate,
        "features": features,
        "classification": classification,
        "parameters": params,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ExoplanetAI")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Home", "Pipeline & Analysis", "Explainability (SHAP)"],
        index=0,
    )

    st.markdown("---")

    # Model status
    clf = st.session_state.classifier
    st.markdown("### Model Status")
    xgb_status = "Loaded" if clf.xgb_model is not None else "Not Found"
    cnn_status = "Loaded" if clf.cnn_model is not None else "Not Found"
    st.markdown(f"**XGBoost**: {xgb_status}")
    st.markdown(f"**CNN**: {cnn_status}")

    if clf.xgb_model is None and clf.cnn_model is None:
        st.warning("No models loaded. Run `python run.py train` first.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#606080; font-size:0.8rem;'>"
        "ExoplanetAI v1.0<br>ISRO Challenge 7</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# Helper: show analysis results inline
# =====================================================================
def _show_analysis_results(result):
    """Display analysis results inline on the current page."""
    st.markdown("---")

    # Classification Result
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

    # Probability bars
    probs = cls.get("probabilities", {})
    if probs:
        prob_df = pd.DataFrame({
            "Class": [k.replace("_", " ").title() for k in probs.keys()],
            "Probability (%)": list(probs.values()),
        })
        fig_prob = px.bar(
            prob_df, x="Probability (%)", y="Class", orientation="h",
            color="Probability (%)",
            color_continuous_scale="Purp",
        )
        fig_prob.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(10,10,30,0.0)",
            plot_bgcolor="rgba(10,10,30,0.3)",
            font=dict(color="#c0c0ff"),
            height=250,
            showlegend=False,
            margin=dict(l=120, r=30, t=30, b=30),
        )
        st.plotly_chart(fig_prob, width='stretch')

    st.markdown("---")

    # Light Curve Plots
    st.markdown("### Light Curves")

    col1, col2 = st.columns(2)

    with col1:
        fig_raw = create_plotly_lightcurve(
            result["time"], result["flux_raw"],
            title="Raw Light Curve", color="#ff6b6b"
        )
        st.plotly_chart(fig_raw, width='stretch')

    with col2:
        fig_clean = create_plotly_lightcurve(
            result["time"], result["flux_flat"],
            title="Cleaned & Detrended", color="#4ecdc4"
        )
        st.plotly_chart(fig_clean, width='stretch')

    # BLS Periodogram
    candidate = result.get("candidate")
    if candidate and candidate.periods is not None:
        st.markdown("### BLS Periodogram")

        fig_bls = go.Figure()
        fig_bls.add_trace(go.Scattergl(
            x=candidate.periods, y=candidate.power,
            mode='lines', line=dict(color="#b794f6", width=1.5),
            name="BLS Power",
        ))
        fig_bls.add_vline(
            x=candidate.period, line_dash="dash",
            line_color="#4ade80",
            annotation_text=f"Best Period: {candidate.period:.4f} d",
            annotation_font_color="#4ade80",
        )
        fig_bls.update_layout(
            title="Box Least Squares Periodogram",
            xaxis_title="Period (days)",
            yaxis_title="BLS Power",
            template="plotly_dark",
            paper_bgcolor="rgba(10,10,30,0.0)",
            plot_bgcolor="rgba(10,10,30,0.3)",
            font=dict(color="#a0a0c0"),
            height=350,
        )
        st.plotly_chart(fig_bls, width='stretch')

    # Phase-Folded Transit
    if candidate and candidate.period > 0:
        st.markdown("### Phase-Folded Transit")

        phase, flux_folded = phase_fold(
            result["time"], result["flux_flat"],
            candidate.period, candidate.t0
        )

        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scattergl(
            x=phase, y=flux_folded,
            mode='markers',
            marker=dict(color="#7c83ff", size=3, opacity=0.5),
            name="Data",
        ))

        # Binned phase curve
        n_bins = 50
        bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        binned_flux = np.array([
            np.median(flux_folded[(phase >= bin_edges[i]) & (phase < bin_edges[i + 1])])
            if ((phase >= bin_edges[i]) & (phase < bin_edges[i + 1])).sum() > 0
            else np.nan
            for i in range(n_bins)
        ])

        fig_phase.add_trace(go.Scatter(
            x=bin_centres, y=binned_flux,
            mode='lines+markers',
            line=dict(color="#fbbf24", width=2.5),
            marker=dict(size=5),
            name="Binned",
        ))

        fig_phase.update_layout(
            title=f"Phase-Folded on P = {candidate.period:.4f} days",
            xaxis_title="Phase",
            yaxis_title="Normalized Flux",
            template="plotly_dark",
            paper_bgcolor="rgba(10,10,30,0.0)",
            plot_bgcolor="rgba(10,10,30,0.3)",
            font=dict(color="#a0a0c0"),
            height=400,
        )
        st.plotly_chart(fig_phase, width='stretch')

    # Transit markers
    if candidate and candidate.period > 0:
        st.markdown("### Detected Transit Events")

        fig_marked = go.Figure()
        fig_marked.add_trace(go.Scatter(
            x=result["time"], y=result["flux_flat"],
            mode='lines', line=dict(color="#4ecdc4", width=1),
            name="Flux",
        ))

        t0 = candidate.t0
        period = candidate.period
        duration = candidate.duration
        t_min, t_max = result["time"].min(), result["time"].max()

        transit_times = []
        t_current = t0
        while t_current < t_max:
            if t_current >= t_min:
                transit_times.append(t_current)
            t_current += period

        for tt in transit_times:
            fig_marked.add_vrect(
                x0=tt - duration / 2, x1=tt + duration / 2,
                fillcolor="rgba(255, 100, 100, 0.15)",
                line_width=0,
            )

        fig_marked.update_layout(
            title="Light Curve with Transit Events Marked",
            xaxis_title="Time (days)",
            yaxis_title="Normalized Flux",
            template="plotly_dark",
            paper_bgcolor="rgba(10,10,30,0.0)",
            plot_bgcolor="rgba(10,10,30,0.3)",
            font=dict(color="#a0a0c0"),
            height=400,
        )
        st.plotly_chart(fig_marked, width='stretch')

    # Parameter Table
    if params:
        st.markdown("### Estimated Parameters")

        param_data = {
            "Parameter": [
                "Orbital Period", "Transit Depth", "Transit Depth",
                "Planet/Star Radius Ratio", "Transit Duration",
                "Impact Parameter", "Number of Transits",
                "Signal-to-Noise Ratio", "Signal Detection Efficiency",
                "Detection Confidence",
            ],
            "Value": [
                f"{params.period_days:.6f} days",
                f"{params.depth_pct:.4f} %",
                f"{params.depth_ppm:.0f} ppm",
                f"{params.radius_ratio:.6f}",
                f"{params.duration_hours:.2f} hours",
                f"{params.impact_parameter:.3f}",
                f"{params.n_transits}",
                f"{params.snr:.2f}",
                f"{params.sde:.2f}",
                f"{params.detection_confidence:.1f} %",
            ],
        }
        st.dataframe(
            pd.DataFrame(param_data),
            width='stretch',
            hide_index=True,
        )

    # Feature Table
    features = result.get("features")
    if features:
        with st.expander("Extracted Features (click to expand)"):
            feat_dict = features.to_dict()
            st.json(feat_dict)


# =====================================================================
# PAGE: HOME
# =====================================================================
if page == "Home":
    st.markdown("# Exoplanet AI Detection Dashboard")
    st.markdown("### AI-Powered Exoplanet Transit Detection from TESS Light Curves")
    st.markdown("---")

    # Stats cards
    n_analyzed = len(st.session_state.analysis_results)
    n_candidates = sum(
        1 for r in st.session_state.analysis_results.values()
        if r.get("parameters") and r["parameters"].is_candidate
    )
    n_batch = len(st.session_state.batch_results)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_analyzed}</div>
            <div class="metric-label">Stars Analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_candidates}</div>
            <div class="metric-label">Transit Candidates</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_batch}</div>
            <div class="metric-label">Batch Results</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        models_count = sum([
            clf.xgb_model is not None,
            clf.cnn_model is not None,
        ])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{models_count}/2</div>
            <div class="metric-label">Models Loaded</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline overview
    st.markdown("### Analysis Pipeline")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        **Data Ingestion** - Upload CSV/FITS light curve data

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

        **Confidence Scoring** - Composite detection confidence
        """)

    # Classification distribution (if results exist)
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("### Classification Distribution")
        classes = [
            r["classification"]["class"]
            for r in st.session_state.analysis_results.values()
            if r.get("classification")
        ]
        if classes:
            class_counts = pd.Series(classes).value_counts()
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                color_discrete_sequence=px.colors.sequential.Purp,
                hole=0.4,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(10,10,30,0.0)",
                font=dict(color="#c0c0ff"),
                height=350,
            )
            st.plotly_chart(fig, width='stretch')

    # Recent results
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("### Recent Analyses")
        rows = []
        for name, r in list(st.session_state.analysis_results.items())[-10:]:
            cls = r.get("classification", {})
            params = r.get("parameters")
            rows.append({
                "File": name,
                "Class": cls.get("class", "-"),
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
        uploaded_file = st.file_uploader(
            "Upload a light curve CSV file",
            type=["csv"],
            help="CSV file with 'time' and 'flux' columns",
        )

        if uploaded_file:
            # Bulletproof CSV loader for messy astronomical files
            try:
                df = pd.read_csv(uploaded_file, on_bad_lines='skip', comment='#')
            except Exception:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=r'\s+', on_bad_lines='skip', comment='#')
                except Exception:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip', comment='#')
                    except Exception:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin1', sep=r'\s+', on_bad_lines='skip', comment='#')

            # Standardize column names (handle uppercase, spaces, and NASA's PDCSAP_FLUX)
            if df is not None and not df.empty:
                df.columns = df.columns.str.strip().str.lower()
                df = df.rename(columns={
                    "pdcsap_flux": "flux", 
                    "sap_flux": "flux", 
                    "bjd": "time", 
                    "jd": "time"
                })

            if df is None or "time" not in df.columns or "flux" not in df.columns:
                st.error("CSV must have 'time' and 'flux' columns!")
            else:
                st.success(f"Loaded **{uploaded_file.name}** - {len(df)} data points")

                # Preview
                fig = create_plotly_lightcurve(
                    df["time"].values, df["flux"].values,
                    title=f"Raw Light Curve: {uploaded_file.name}"
                )
                st.plotly_chart(fig, width='stretch')

                if st.button("Run Full Analysis", type="primary", key="analyze_upload"):
                    result = run_full_analysis(
                        df["time"].values, df["flux"].values,
                        filename=uploaded_file.name
                    )
                    st.session_state.analysis_results[uploaded_file.name] = result
                    st.session_state.uploaded_data[uploaded_file.name] = df
                    st.success("Analysis complete! See results below.")

                    # Show results inline
                    _show_analysis_results(result)

    with tab2:
        samples = load_sample_data()
        if not samples:
            st.warning("No synthetic data found. Run `python run.py generate` first.")
        else:
            st.markdown("### Select a sample light curve:")
            sample_class = st.selectbox(
                "Signal Class",
                list(samples.keys()),
                format_func=lambda x: {
                    "transit": "Exoplanet Transit",
                    "eclipsing_binary": "Eclipsing Binary",
                    "variable_star": "Variable Star",
                    "blend": "Blend",
                    "noise": "Noise",
                }.get(x, x)
            )

            sample_files = samples[sample_class]
            sample_file = st.selectbox(
                "Select file",
                sample_files,
                format_func=lambda x: x.name,
            )

            if sample_file:
                try:
                    df = pd.read_csv(sample_file, on_bad_lines='skip', comment='#')
                except Exception:
                    try:
                        df = pd.read_csv(sample_file, sep=r'\s+', on_bad_lines='skip', comment='#')
                    except Exception:
                        try:
                            df = pd.read_csv(sample_file, encoding='latin1', on_bad_lines='skip', comment='#')
                        except Exception:
                            df = pd.read_csv(sample_file, encoding='latin1', sep=r'\s+', on_bad_lines='skip', comment='#')
                
                # Standardize column names
                if df is not None and not df.empty:
                    df.columns = df.columns.str.strip().str.lower()
                    df = df.rename(columns={
                        "pdcsap_flux": "flux", 
                        "sap_flux": "flux", 
                        "bjd": "time", 
                        "jd": "time"
                    })

                st.info(f"{sample_file.name} - {len(df)} points - Class: **{sample_class}**")

                fig = create_plotly_lightcurve(
                    df["time"].values, df["flux"].values,
                    title=f"Sample: {sample_file.name} ({sample_class})"
                )
                st.plotly_chart(fig, width='stretch')

                if st.button("Run Full Analysis", type="primary", key="analyze_sample"):
                    result = run_full_analysis(
                        df["time"].values, df["flux"].values,
                        filename=sample_file.name
                    )
                    st.session_state.analysis_results[sample_file.name] = result
                    st.success("Analysis complete! See results below.")

                    # Show results inline
                    _show_analysis_results(result)



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
        selected = st.selectbox("Select analysis:", analysis_names,
                                 index=len(analysis_names) - 1)
        result = st.session_state.analysis_results[selected]

        explanation = result.get("explanation")

        if explanation is None:
            st.warning("No explainability data. Models may not be loaded.")
        else:
            # Check if physics overruled the AI
            predicted_class = result["classification"]["class"]
            is_candidate = result.get("parameters") and result["parameters"].is_candidate
            
            # Natural language explanation
            st.markdown("### Explanation")
            
            if predicted_class == "transit" and not is_candidate:
                st.warning("⚠️ **Note:** The AI model predicted this was a Transit based on the features below, but the Physics module ultimately rejected it because the physical parameters (like planetary radius) were impossible for a planet. The SHAP chart below explains why the AI was fooled into thinking it was a Transit.")

            st.markdown(f"""
            <div class="result-card">
                <pre style="color: #c0c0ff; white-space: pre-wrap; font-family: Inter, sans-serif;">
{explanation.get('explanation_text', 'No explanation available.')}
                </pre>
                <p style="color: #606080; font-size: 0.8rem;">Method: {explanation.get('method', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

            # Feature contributions chart
            contributions = explanation.get("contributions", [])
            if contributions:
                st.markdown("### Feature Contributions")

                feat_names = [c["feature"].replace("_", " ").title() for c in contributions[:10]]
                feat_importance = [c.get("importance_pct", abs(c.get("shap_value", 0)) * 100) for c in contributions[:10]]
                feat_colors = [
                    "#4ade80" if c.get("impact") == "positive" or c.get("shap_value", 0) > 0
                    else "#f87171"
                    for c in contributions[:10]
                ]

                fig_shap = go.Figure()
                fig_shap.add_trace(go.Bar(
                    y=feat_names[::-1],
                    x=feat_importance[::-1],
                    orientation='h',
                    marker_color=feat_colors[::-1],
                ))
                fig_shap.update_layout(
                    title="Feature Importance (Top 10)",
                    xaxis_title="Importance (%)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(10,10,30,0.0)",
                    plot_bgcolor="rgba(10,10,30,0.3)",
                    font=dict(color="#c0c0ff"),
                    height=400,
                    margin=dict(l=160, r=30, t=50, b=40),
                )
                st.plotly_chart(fig_shap, width='stretch')

            # Global feature importance
            clf = st.session_state.classifier
            if clf.xgb_model is not None:
                st.markdown("### Global Feature Importance")
                global_imp = get_global_importance(clf.xgb_model)
                if global_imp:
                    g_names = list(global_imp.keys())[:10]
                    g_vals = list(global_imp.values())[:10]

                    fig_global = px.bar(
                        x=g_vals[::-1],
                        y=[n.replace("_", " ").title() for n in g_names[::-1]],
                        orientation='h',
                        color=g_vals[::-1],
                        color_continuous_scale="Purp",
                    )
                    fig_global.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(10,10,30,0.0)",
                        plot_bgcolor="rgba(10,10,30,0.3)",
                        font=dict(color="#c0c0ff"),
                        height=380,
                        showlegend=False,
                        margin=dict(l=160, r=30, t=30, b=40),
                        xaxis_title="Importance",
                    )
                    st.plotly_chart(fig_global, width='stretch')
