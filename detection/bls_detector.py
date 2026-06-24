"""
Box Least Squares (BLS) Transit Detection Engine

Uses astropy's BoxLeastSquares to detect periodic transit-like signals
in cleaned light curves.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from astropy.timeseries import BoxLeastSquares
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


@dataclass
class TransitCandidate:
    """Container for a detected transit candidate."""
    period: float          # Best-fit period (days)
    depth: float           # Transit depth (fractional)
    duration: float        # Transit duration (days)
    t0: float              # Transit midpoint (days)
    bls_power: float       # BLS power at best period
    sde: float             # Signal Detection Efficiency
    snr: float             # Signal-to-noise ratio
    n_transits: int        # Number of transits in the data
    periods: Optional[np.ndarray] = field(default=None, repr=False)
    power: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "period": round(self.period, 6),
            "depth": round(self.depth, 8),
            "depth_pct": round(self.depth * 100, 4),
            "duration_hours": round(self.duration * 24, 2),
            "t0": round(self.t0, 4),
            "bls_power": round(self.bls_power, 6),
            "sde": round(self.sde, 2),
            "snr": round(self.snr, 2),
            "n_transits": self.n_transits,
        }


def run_bls(time: np.ndarray, flux: np.ndarray,
            min_period: float = 0.5,
            max_period: float = 20.0,
            n_periods: int = 500,
            min_duration: float = 0.01,
            max_duration: float = 0.25,
            n_durations: int = 10) -> TransitCandidate | None:
    """
    Run Box Least Squares transit detection.

    Parameters
    ----------
    time : np.ndarray
        Time array (days).
    flux : np.ndarray
        Normalized flux array (centered at ~1.0).
    min_period, max_period : float
        Period search range (days).
    n_periods : int
        Number of trial periods.
    min_duration, max_duration : float
        Transit duration range as fraction of period.
    n_durations : int
        Number of trial durations.

    Returns
    -------
    candidate : TransitCandidate or None
        Detected transit candidate, or None if BLS is unavailable.
    """
    if not HAS_ASTROPY:
        return _fallback_bls(time, flux, min_period, max_period, n_periods)

    # Remove NaNs
    mask = np.isfinite(time) & np.isfinite(flux)
    t, f = time[mask], flux[mask]

    if len(t) < 100:
        return None

    # Build BLS model
    bls = BoxLeastSquares(t, f)

    # Compute periodogram
    durations = np.linspace(min_duration, max_duration, n_durations)
    periods = np.linspace(min_period, max_period, n_periods)

    result = bls.power(periods, durations)

    # Find best period
    best_idx = np.argmax(result.power)
    best_period = result.period[best_idx]
    best_power = result.power[best_idx]
    best_duration = result.duration[best_idx]
    best_t0 = result.transit_time[best_idx]
    best_depth = result.depth[best_idx]

    # Compute SDE (Signal Detection Efficiency)
    power_mean = np.mean(result.power)
    power_std = np.std(result.power)
    sde = (best_power - power_mean) / power_std if power_std > 0 else 0

    # Estimate SNR
    snr = _estimate_transit_snr(t, f, best_period, best_duration, best_t0)

    # Count transits
    data_span = t[-1] - t[0]
    n_transits = max(1, int(data_span / best_period))

    return TransitCandidate(
        period=float(best_period),
        depth=float(best_depth),
        duration=float(best_duration),
        t0=float(best_t0),
        bls_power=float(best_power),
        sde=float(sde),
        snr=float(snr),
        n_transits=n_transits,
        periods=np.array(result.period),
        power=np.array(result.power),
    )


def _fallback_bls(time, flux, min_period, max_period, n_periods):
    """Simple fallback BLS when astropy is not available."""
    mask = np.isfinite(time) & np.isfinite(flux)
    t, f = time[mask], flux[mask]

    if len(t) < 100:
        return None

    best_power = -np.inf
    best_period = min_period
    best_depth = 0.0
    best_duration = 0.0
    best_t0 = 0.0

    periods = np.linspace(min_period, max_period, min(n_periods, 2000))
    powers = np.zeros(len(periods))

    for idx, period in enumerate(periods):
        phase = (t % period) / period

        for dur_frac in [0.02, 0.05, 0.1]:
            in_transit = phase < dur_frac
            if in_transit.sum() < 3 or (~in_transit).sum() < 3:
                continue

            depth = np.median(f[~in_transit]) - np.median(f[in_transit])
            if depth <= 0:
                continue

            residuals = f.copy()
            residuals[in_transit] += depth
            power = 1.0 / np.std(residuals)

            if power > best_power:
                best_power = power
                best_period = period
                best_depth = depth
                best_duration = dur_frac * period
                best_t0 = 0.0

        powers[idx] = best_power if best_power > 0 else 0

    power_mean = np.mean(powers[powers > 0]) if (powers > 0).any() else 0
    power_std = np.std(powers[powers > 0]) if (powers > 0).any() else 1
    sde = (best_power - power_mean) / power_std if power_std > 0 else 0

    snr = _estimate_transit_snr(t, f, best_period, best_duration, best_t0)
    n_transits = max(1, int((t[-1] - t[0]) / best_period))

    return TransitCandidate(
        period=float(best_period),
        depth=float(best_depth),
        duration=float(best_duration),
        t0=float(best_t0),
        bls_power=float(best_power),
        sde=float(sde),
        snr=float(snr),
        n_transits=n_transits,
        periods=periods,
        power=powers,
    )


def _estimate_transit_snr(time, flux, period, duration, t0):
    """Estimate transit SNR from in-transit vs out-of-transit scatter."""
    phase = ((time - t0) % period) / period
    half_dur = (duration / period) / 2

    in_transit = np.abs(phase) < half_dur
    out_transit = ~in_transit

    if in_transit.sum() < 3 or out_transit.sum() < 3:
        return 0.0

    depth = np.median(flux[out_transit]) - np.median(flux[in_transit])
    scatter = np.std(flux[out_transit])

    if scatter == 0:
        return 0.0

    return depth / scatter * np.sqrt(in_transit.sum())


def phase_fold(time: np.ndarray, flux: np.ndarray,
               period: float, t0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase-fold a light curve on a given period.

    Returns
    -------
    phase : np.ndarray
        Phase values in [-0.5, 0.5].
    flux_sorted : np.ndarray
        Flux sorted by phase.
    """
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]
