"""
Feature Extractor for ExoplanetAI

Extracts 12+ ML-ready features from processed light curves
and BLS transit detection results.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detection.bls_detector import run_bls, phase_fold, TransitCandidate


@dataclass
class LightCurveFeatures:
    """Container for extracted features."""
    transit_depth: float        # Fractional brightness decrease
    transit_duration: float     # Duration in hours
    orbital_period: float       # Period in days
    snr: float                  # Signal-to-noise ratio
    transit_symmetry: float     # Ingress/egress symmetry (1.0 = perfect)
    n_transits: int             # Number of detected transits
    bls_power: float            # BLS periodogram peak
    sde: float                  # Signal Detection Efficiency
    depth_even_odd: float       # Even/odd transit depth ratio
    secondary_depth: float      # Secondary eclipse depth (binary indicator)
    flux_std: float             # Overall flux standard deviation
    flux_skewness: float        # Flux distribution skewness
    flux_kurtosis: float        # Flux distribution kurtosis
    amplitude: float            # Peak-to-peak flux amplitude

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        return np.array([
            self.transit_depth,
            self.transit_duration,
            self.orbital_period,
            self.snr,
            self.transit_symmetry,
            self.n_transits,
            self.bls_power,
            self.sde,
            self.depth_even_odd,
            self.secondary_depth,
            self.flux_std,
            self.flux_skewness,
            self.flux_kurtosis,
            self.amplitude,
        ])

    def to_dict(self) -> dict:
        return {
            "transit_depth": round(self.transit_depth, 8),
            "transit_depth_pct": round(self.transit_depth * 100, 4),
            "transit_duration_hours": round(self.transit_duration, 2),
            "orbital_period_days": round(self.orbital_period, 4),
            "snr": round(self.snr, 2),
            "transit_symmetry": round(self.transit_symmetry, 4),
            "n_transits": self.n_transits,
            "bls_power": round(self.bls_power, 6),
            "sde": round(self.sde, 2),
            "depth_even_odd": round(self.depth_even_odd, 4),
            "secondary_depth": round(self.secondary_depth, 8),
            "flux_std": round(self.flux_std, 6),
            "flux_skewness": round(self.flux_skewness, 4),
            "flux_kurtosis": round(self.flux_kurtosis, 4),
            "amplitude": round(self.amplitude, 6),
        }

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "transit_depth", "transit_duration", "orbital_period",
            "snr", "transit_symmetry", "n_transits", "bls_power",
            "sde", "depth_even_odd", "secondary_depth", "flux_std",
            "flux_skewness", "flux_kurtosis", "amplitude",
        ]


def extract_features(time: np.ndarray, flux: np.ndarray,
                     candidate: TransitCandidate = None) -> LightCurveFeatures:
    """
    Extract features from a light curve.

    Parameters
    ----------
    time : np.ndarray
        Time array (days).
    flux : np.ndarray
        Cleaned, detrended flux array.
    candidate : TransitCandidate, optional
        Pre-computed BLS result. If None, BLS is run internally.

    Returns
    -------
    features : LightCurveFeatures
    """
    mask = np.isfinite(time) & np.isfinite(flux)
    t, f = time[mask], flux[mask]

    # Run BLS if no candidate provided
    if candidate is None:
        candidate = run_bls(t, f)

    # --- BLS-based features ---
    if candidate is not None:
        transit_depth = candidate.depth
        transit_duration = candidate.duration * 24  # hours
        orbital_period = candidate.period
        snr = candidate.snr
        bls_power = candidate.bls_power
        sde = candidate.sde
        n_transits = candidate.n_transits
        t0 = candidate.t0
    else:
        transit_depth = 0.0
        transit_duration = 0.0
        orbital_period = 0.0
        snr = 0.0
        bls_power = 0.0
        sde = 0.0
        n_transits = 0
        t0 = 0.0

    # --- Transit symmetry ---
    symmetry = _compute_symmetry(t, f, orbital_period, candidate.duration if candidate else 0, t0)

    # --- Even/odd depth ratio ---
    depth_ratio = _even_odd_depth(t, f, orbital_period, candidate.duration if candidate else 0, t0)

    # --- Secondary eclipse depth ---
    secondary = _secondary_eclipse_depth(t, f, orbital_period, candidate.duration if candidate else 0, t0)

    # --- Statistical features ---
    flux_std = float(np.std(f))
    flux_skew = float(skew(f))
    flux_kurt = float(kurtosis(f))
    amplitude = float(np.ptp(f))  # peak-to-peak

    return LightCurveFeatures(
        transit_depth=transit_depth,
        transit_duration=transit_duration,
        orbital_period=orbital_period,
        snr=snr,
        transit_symmetry=symmetry,
        n_transits=n_transits,
        bls_power=bls_power,
        sde=sde,
        depth_even_odd=depth_ratio,
        secondary_depth=secondary,
        flux_std=flux_std,
        flux_skewness=flux_skew,
        flux_kurtosis=flux_kurt,
        amplitude=amplitude,
    )


def _compute_symmetry(time, flux, period, duration, t0):
    """Compute transit ingress/egress symmetry (1.0 = perfectly symmetric)."""
    if period <= 0 or duration <= 0:
        return 0.0

    phase, f_folded = phase_fold(time, flux, period, t0)
    half_dur = (duration / period) / 2

    # Ingress: just before transit
    ingress_mask = (phase > -half_dur * 1.5) & (phase < -half_dur * 0.5)
    # Egress: just after transit
    egress_mask = (phase > half_dur * 0.5) & (phase < half_dur * 1.5)

    if ingress_mask.sum() < 3 or egress_mask.sum() < 3:
        return 0.5

    ingress_slope = np.abs(np.mean(np.diff(f_folded[ingress_mask])))
    egress_slope = np.abs(np.mean(np.diff(f_folded[egress_mask])))

    if max(ingress_slope, egress_slope) == 0:
        return 1.0

    return min(ingress_slope, egress_slope) / max(ingress_slope, egress_slope)


def _even_odd_depth(time, flux, period, duration, t0):
    """Compare depths of even vs odd transits (different depths → binary)."""
    if period <= 0 or duration <= 0:
        return 1.0

    transit_number = np.floor((time - t0) / period)
    phase = ((time - t0) % period) / period
    half_dur = (duration / period) / 2
    in_transit = np.abs(phase) < half_dur

    out_median = np.median(flux[~in_transit]) if (~in_transit).any() else 1.0

    even_mask = in_transit & (transit_number.astype(int) % 2 == 0)
    odd_mask = in_transit & (transit_number.astype(int) % 2 == 1)

    even_depth = out_median - np.median(flux[even_mask]) if even_mask.sum() > 2 else 0
    odd_depth = out_median - np.median(flux[odd_mask]) if odd_mask.sum() > 2 else 0

    if max(abs(even_depth), abs(odd_depth)) == 0:
        return 1.0

    return min(abs(even_depth), abs(odd_depth)) / max(abs(even_depth), abs(odd_depth))


def _secondary_eclipse_depth(time, flux, period, duration, t0):
    """Check for secondary eclipse at phase 0.5 (binary star indicator)."""
    if period <= 0 or duration <= 0:
        return 0.0

    phase = ((time - t0) % period) / period
    half_dur = (duration / period) / 2

    # Secondary eclipse at phase ~0.5
    secondary_mask = np.abs(phase - 0.5) < half_dur
    out_mask = (np.abs(phase) > half_dur * 2) & (np.abs(phase - 0.5) > half_dur * 2)

    if secondary_mask.sum() < 3 or out_mask.sum() < 3:
        return 0.0

    return float(np.median(flux[out_mask]) - np.median(flux[secondary_mask]))


def batch_extract(time_list: list, flux_list: list) -> np.ndarray:
    """Extract features from multiple light curves. Returns feature matrix."""
    features = []
    for t, f in zip(time_list, flux_list):
        feat = extract_features(t, f)
        features.append(feat.to_array())
    return np.array(features)
