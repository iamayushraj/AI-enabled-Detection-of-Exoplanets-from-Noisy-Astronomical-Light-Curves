"""
Parameter Estimator for ExoplanetAI

Estimates and refines orbital parameters from transit detections:
- Orbital period
- Transit depth → planet/star radius ratio
- Transit duration
- Impact parameter
- Detection confidence score
"""

import numpy as np
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detection.bls_detector import TransitCandidate, phase_fold


@dataclass
class PlanetaryParameters:
    """Estimated planetary / orbital parameters."""
    period_days: float              # Orbital period
    period_uncertainty: float       # Period uncertainty
    depth_pct: float                # Transit depth in percent
    depth_ppm: float                # Transit depth in parts per million
    radius_ratio: float             # Rp/Rs (planet/star radius ratio)
    duration_hours: float           # Transit duration
    impact_parameter: float         # Impact parameter b (0 = central, 1 = grazing)
    n_transits: int                 # Number of observed transits
    snr: float                      # Signal-to-noise ratio
    sde: float                      # Signal Detection Efficiency
    detection_confidence: float     # Composite confidence (0-100%)
    is_candidate: bool              # Whether it passes detection threshold

    def to_dict(self) -> dict:
        return {
            "period_days": round(self.period_days, 6),
            "period_uncertainty_days": round(self.period_uncertainty, 6),
            "depth_pct": round(self.depth_pct, 4),
            "depth_ppm": round(self.depth_ppm, 1),
            "radius_ratio_Rp_Rs": round(self.radius_ratio, 6),
            "duration_hours": round(self.duration_hours, 2),
            "impact_parameter": round(self.impact_parameter, 3),
            "n_transits": self.n_transits,
            "snr": round(self.snr, 2),
            "sde": round(self.sde, 2),
            "detection_confidence_pct": round(self.detection_confidence, 2),
            "is_candidate": self.is_candidate,
        }

    def summary_text(self) -> str:
        """Human-readable summary."""
        status = "✅ TRANSIT CANDIDATE" if self.is_candidate else "❌ NOT A CANDIDATE"
        return (
            f"\n{'='*50}\n"
            f"  {status}\n"
            f"{'='*50}\n"
            f"  Period         : {self.period_days:.4f} ± {self.period_uncertainty:.4f} days\n"
            f"  Depth          : {self.depth_pct:.4f}% ({self.depth_ppm:.0f} ppm)\n"
            f"  Radius Ratio   : {self.radius_ratio:.4f} Rp/Rs\n"
            f"  Duration       : {self.duration_hours:.2f} hours\n"
            f"  Impact Param.  : {self.impact_parameter:.3f}\n"
            f"  Transits       : {self.n_transits}\n"
            f"  SNR            : {self.snr:.1f}\n"
            f"  SDE            : {self.sde:.1f}\n"
            f"  Confidence     : {self.detection_confidence:.1f}%\n"
            f"{'='*50}\n"
        )


def estimate_parameters(time: np.ndarray, flux: np.ndarray,
                        candidate: TransitCandidate,
                        classification: dict = None) -> PlanetaryParameters:
    """
    Estimate planetary parameters from a transit candidate.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    flux : np.ndarray
        Cleaned flux array.
    candidate : TransitCandidate
        BLS detection result.
    classification : dict, optional
        Classification result (used for confidence weighting).

    Returns
    -------
    params : PlanetaryParameters
    """
    # --- Period refinement ---
    period = candidate.period
    period_unc = _estimate_period_uncertainty(time, flux, period, candidate.t0)

    # --- Depth ---
    depth_frac = max(candidate.depth, 0.0)
    depth_pct = depth_frac * 100
    depth_ppm = depth_frac * 1e6

    # --- Radius ratio: Rp/Rs = sqrt(depth) ---
    radius_ratio = np.sqrt(depth_frac) if depth_frac > 0 else 0.0

    # --- Duration ---
    duration_hours = candidate.duration * 24

    # --- Impact parameter estimate ---
    # b ≈ sqrt(1 - (duration/period * π)^2 * ... )
    # Simplified: use depth shape
    impact_param = _estimate_impact_parameter(time, flux, candidate)

    # --- Composite confidence ---
    confidence = _compute_confidence(candidate, classification)

    # --- Is it a candidate? ---
    is_candidate = (
        candidate.sde > 6.0 and
        candidate.snr > 3.0 and
        candidate.depth > 0.0001 and
        confidence > 50.0
    )

    return PlanetaryParameters(
        period_days=period,
        period_uncertainty=period_unc,
        depth_pct=depth_pct,
        depth_ppm=depth_ppm,
        radius_ratio=radius_ratio,
        duration_hours=duration_hours,
        impact_parameter=impact_param,
        n_transits=candidate.n_transits,
        snr=candidate.snr,
        sde=candidate.sde,
        detection_confidence=confidence,
        is_candidate=is_candidate,
    )


def _estimate_period_uncertainty(time, flux, period, t0):
    """Estimate period uncertainty from transit timing scatter."""
    if period <= 0:
        return 0.0

    mask = np.isfinite(time) & np.isfinite(flux)
    t, f = time[mask], flux[mask]
    data_span = t[-1] - t[0]

    # Rough estimate: period / (n_transits * SNR)
    n_transits = max(1, int(data_span / period))

    # Phase residuals
    phase = ((t - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    # Transit points
    in_transit = np.abs(phase) < 0.05
    if in_transit.sum() < 5:
        return period * 0.01

    # Timing precision ≈ period / (n_transits^1.5)
    uncertainty = period / (n_transits ** 1.5) * 0.1
    return max(uncertainty, period * 1e-5)


def _estimate_impact_parameter(time, flux, candidate):
    """Estimate impact parameter from transit shape (0=central, 1=grazing)."""
    if candidate.period <= 0 or candidate.duration <= 0:
        return 0.5

    phase, f_folded = phase_fold(time, flux, candidate.period, candidate.t0)
    half_dur = (candidate.duration / candidate.period) / 2

    in_transit = np.abs(phase) < half_dur
    if in_transit.sum() < 5:
        return 0.5

    transit_flux = f_folded[in_transit]

    # Flat-bottomed transit → low impact parameter (central)
    # V-shaped transit → high impact parameter (grazing)
    if len(transit_flux) < 5:
        return 0.5

    # Compare depth at centre vs edges of transit
    n = len(transit_flux)
    centre = transit_flux[n // 4: 3 * n // 4]
    edges = np.concatenate([transit_flux[: n // 4], transit_flux[3 * n // 4:]])

    if len(centre) < 2 or len(edges) < 2:
        return 0.5

    centre_depth = 1.0 - np.median(centre)
    edge_depth = 1.0 - np.median(edges)

    if centre_depth == 0:
        return 0.5

    flatness = edge_depth / centre_depth  # ~1 = flat bottom, <1 = V-shaped
    impact = max(0, min(1, 1.0 - flatness))

    return float(impact)


def _compute_confidence(candidate, classification=None):
    """Compute composite detection confidence (0-100%)."""
    score = 0.0

    # SDE contribution (0-40 points)
    sde_score = min(candidate.sde / 15.0, 1.0) * 40
    score += sde_score

    # SNR contribution (0-25 points)
    snr_score = min(candidate.snr / 20.0, 1.0) * 25
    score += snr_score

    # Multiple transits bonus (0-15 points)
    transit_score = min(candidate.n_transits / 5.0, 1.0) * 15
    score += transit_score

    # Depth sanity (0-10 points)
    if 0.0001 < candidate.depth < 0.05:
        score += 10  # Planet-like depth range
    elif candidate.depth >= 0.05:
        score += 3   # Could be binary

    # Classification confidence (0-10 points)
    if classification and classification.get("class") == "transit":
        cls_conf = classification.get("confidence", 0) / 100.0
        score += cls_conf * 10

    return min(score, 100.0)
