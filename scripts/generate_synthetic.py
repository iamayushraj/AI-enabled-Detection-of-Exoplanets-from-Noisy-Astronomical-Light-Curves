"""
Synthetic Light Curve Generator for ExoplanetAI

Generates realistic synthetic TESS-like light curves for 5 classes:
- Transit: Periodic box-shaped dips with limb darkening
- Eclipsing Binary: Deep, V-shaped dips with secondary eclipses
- Stellar Variability: Sinusoidal variations
- Blend: Weak transit mixed with variability
- Noise: Pure instrumental noise
"""

import numpy as np
import pandas as pd
import os
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TESS_SECTOR_DAYS = 27.4          # One TESS sector duration
CADENCE_MINUTES = 2.0            # TESS short cadence
N_POINTS = int(TESS_SECTOR_DAYS * 24 * 60 / CADENCE_MINUTES)  # ~19728
CLASSES = ["transit", "eclipsing_binary", "variable_star", "blend", "noise"]


# ---------------------------------------------------------------------------
# Helper: add realistic noise
# ---------------------------------------------------------------------------
def _add_noise(flux: np.ndarray, noise_level: float = 0.001,
               red_noise_amp: float = 0.0003) -> np.ndarray:
    """Add white (Gaussian) + correlated red noise to flux array."""
    white = np.random.normal(0, noise_level, len(flux))

    # Red noise via random walk
    red = np.cumsum(np.random.normal(0, red_noise_amp / np.sqrt(len(flux)), len(flux)))
    red -= np.polyval(np.polyfit(np.arange(len(flux)), red, 1), np.arange(len(flux)))

    # Occasional outliers (~0.5 % of points)
    outlier_mask = np.random.random(len(flux)) < 0.005
    outliers = np.zeros(len(flux))
    outliers[outlier_mask] = np.random.normal(0, noise_level * 5, outlier_mask.sum())

    return flux + white + red + outliers


# ---------------------------------------------------------------------------
# Limb-darkened transit model  (quadratic limb darkening)
# ---------------------------------------------------------------------------
def _transit_model(time: np.ndarray, period: float, depth: float,
                   duration_hours: float, t0: float) -> np.ndarray:
    """Generate a simplified transit model with smooth ingress/egress."""
    flux = np.ones_like(time)
    duration_days = duration_hours / 24.0
    half_dur = duration_days / 2.0
    ingress = duration_days * 0.15          # ingress/egress fraction

    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0               # centre on 0

    phase_dur = half_dur / period

    for i in range(len(flux)):
        p = abs(phase[i])
        if p < phase_dur - ingress / period:
            flux[i] = 1.0 - depth
        elif p < phase_dur:
            # smooth ingress/egress (cosine taper)
            frac = (p - (phase_dur - ingress / period)) / (ingress / period)
            flux[i] = 1.0 - depth * (1.0 - frac)
    return flux


# ---------------------------------------------------------------------------
# Generators per class
# ---------------------------------------------------------------------------
def generate_transit(time: np.ndarray, rng: np.random.Generator) -> dict:
    """Planetary transit: small depth, periodic, symmetric."""
    period = rng.uniform(1.0, 15.0)
    depth = rng.uniform(0.0005, 0.025)       # 0.05 % – 2.5 %
    duration = rng.uniform(1.0, 5.0)         # hours
    t0 = rng.uniform(0, period)
    flux = _transit_model(time, period, depth, duration, t0)
    flux = _add_noise(flux, noise_level=rng.uniform(0.0005, 0.002))
    return {
        "flux": flux,
        "params": {"period": round(period, 4), "depth": round(depth, 6),
                    "duration_hours": round(duration, 2), "t0": round(t0, 4)},
    }


def generate_eclipsing_binary(time: np.ndarray, rng: np.random.Generator) -> dict:
    """Eclipsing binary: deep primary + shallower secondary eclipse."""
    period = rng.uniform(0.5, 10.0)
    primary_depth = rng.uniform(0.05, 0.50)   # 5 – 50 %
    secondary_depth = primary_depth * rng.uniform(0.1, 0.6)
    duration = rng.uniform(1.5, 6.0)
    t0 = rng.uniform(0, period)

    flux = _transit_model(time, period, primary_depth, duration, t0)
    # Add secondary eclipse at phase 0.5
    flux2 = _transit_model(time, period, secondary_depth, duration * 0.8, t0 + period / 2)
    flux = flux * flux2
    flux = _add_noise(flux, noise_level=rng.uniform(0.001, 0.003))
    return {
        "flux": flux,
        "params": {"period": round(period, 4),
                    "primary_depth": round(primary_depth, 4),
                    "secondary_depth": round(secondary_depth, 4),
                    "duration_hours": round(duration, 2)},
    }


def generate_variable_star(time: np.ndarray, rng: np.random.Generator) -> dict:
    """Stellar variability: multi-frequency sinusoidal brightness changes."""
    n_modes = rng.integers(1, 4)
    flux = np.ones_like(time)
    for _ in range(n_modes):
        amp = rng.uniform(0.002, 0.03)
        freq = rng.uniform(0.1, 3.0)        # cycles / day
        phi = rng.uniform(0, 2 * np.pi)
        flux += amp * np.sin(2 * np.pi * freq * time + phi)
    flux = _add_noise(flux, noise_level=rng.uniform(0.0005, 0.002))
    return {"flux": flux, "params": {"n_modes": int(n_modes)}}


def generate_blend(time: np.ndarray, rng: np.random.Generator) -> dict:
    """Blend: weak transit-like dip + stellar variability contamination."""
    # Weak transit
    period = rng.uniform(2.0, 12.0)
    depth = rng.uniform(0.0003, 0.005)
    duration = rng.uniform(1.5, 4.0)
    t0 = rng.uniform(0, period)
    flux = _transit_model(time, period, depth, duration, t0)
    # Add variability
    amp = rng.uniform(0.003, 0.015)
    freq = rng.uniform(0.2, 2.0)
    flux += amp * np.sin(2 * np.pi * freq * time + rng.uniform(0, 2 * np.pi))
    flux = _add_noise(flux, noise_level=rng.uniform(0.001, 0.003))
    return {
        "flux": flux,
        "params": {"period": round(period, 4), "depth": round(depth, 6),
                    "variability_amp": round(amp, 4)},
    }


def generate_noise(time: np.ndarray, rng: np.random.Generator) -> dict:
    """Pure noise: no astrophysical signal."""
    flux = np.ones_like(time)
    flux = _add_noise(flux, noise_level=rng.uniform(0.001, 0.004),
                      red_noise_amp=rng.uniform(0.0003, 0.001))
    # Occasional systematic trend
    if rng.random() < 0.3:
        flux += 0.002 * np.sin(2 * np.pi * time / TESS_SECTOR_DAYS)
    return {"flux": flux, "params": {}}


# ---------------------------------------------------------------------------
# Class → generator map
# ---------------------------------------------------------------------------
GENERATORS = {
    "transit": generate_transit,
    "eclipsing_binary": generate_eclipsing_binary,
    "variable_star": generate_variable_star,
    "blend": generate_blend,
    "noise": generate_noise,
}


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_dataset(output_dir: str, n_per_class: int = 100,
                     n_points: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a full synthetic dataset.

    Parameters
    ----------
    output_dir : str
        Directory to save CSV files and labels.
    n_per_class : int
        Number of light curves per class.
    n_points : int
        Number of time points per light curve (down-sampled from full TESS).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    labels_df : pd.DataFrame
        DataFrame with columns [filename, label, params].
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    time = np.linspace(0, TESS_SECTOR_DAYS, n_points)
    records = []

    for cls in CLASSES:
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        gen = GENERATORS[cls]

        for i in range(n_per_class):
            result = gen(time, rng)
            fname = f"{cls}_{i:04d}.csv"
            fpath = os.path.join(cls_dir, fname)

            df = pd.DataFrame({"time": time, "flux": result["flux"]})
            df.to_csv(fpath, index=False)

            records.append({
                "filename": f"{cls}/{fname}",
                "label": cls,
                "params": json.dumps(result["params"]),
            })

    labels_df = pd.DataFrame(records)
    labels_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)
    print(f"✅ Generated {len(records)} light curves in {output_dir}")
    print(f"   Classes: { {c: n_per_class for c in CLASSES} }")
    return labels_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "synthetic"
    generate_dataset(str(data_dir), n_per_class=100, n_points=2000)
