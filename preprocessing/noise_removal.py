"""
Noise Removal Module for ExoplanetAI

Implements multiple noise reduction techniques for stellar light curves:
- Sigma clipping (outlier removal)
- Savitzky-Golay smoothing
- Median filtering (cosmic ray spike removal)
- NaN/gap interpolation
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.stats import median_abs_deviation


def sigma_clip(flux: np.ndarray, sigma: float = 3.0,
               max_iters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively remove outliers beyond sigma * MAD from the median.

    Parameters
    ----------
    flux : np.ndarray
        Input flux array.
    sigma : float
        Number of sigma for clipping threshold.
    max_iters : int
        Maximum clipping iterations.

    Returns
    -------
    cleaned_flux : np.ndarray
        Flux with outliers replaced by NaN.
    mask : np.ndarray
        Boolean mask (True = kept, False = clipped).
    """
    cleaned = flux.copy()
    mask = np.ones(len(flux), dtype=bool)

    for _ in range(max_iters):
        valid = cleaned[mask & np.isfinite(cleaned)]
        if len(valid) == 0:
            break
        med = np.median(valid)
        mad = median_abs_deviation(valid)
        if mad == 0:
            break

        threshold = sigma * mad * 1.4826  # convert MAD to std equivalent
        new_mask = np.abs(cleaned - med) <= threshold
        new_mask &= np.isfinite(cleaned)

        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    cleaned[~mask] = np.nan
    return cleaned, mask


def savgol_smooth(flux: np.ndarray, window_length: int = 51,
                  polyorder: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for smoothing while preserving transit shape.

    Parameters
    ----------
    flux : np.ndarray
        Input flux (NaN values are interpolated before filtering).
    window_length : int
        Filter window size (must be odd).
    polyorder : int
        Polynomial order for local fits.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed flux array.
    """
    # Interpolate NaNs for filtering
    clean = _interpolate_nans(flux)

    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(clean):
        window_length = len(clean) // 2 * 2 + 1

    return savgol_filter(clean, window_length, polyorder)


def median_filter(flux: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter to remove sharp spikes (cosmic rays, etc.).

    Parameters
    ----------
    flux : np.ndarray
        Input flux array.
    kernel_size : int
        Median filter kernel size (must be odd).

    Returns
    -------
    filtered : np.ndarray
        Median-filtered flux.
    """
    clean = _interpolate_nans(flux)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(clean, kernel_size)


def _interpolate_nans(flux: np.ndarray) -> np.ndarray:
    """Replace NaN values with linear interpolation."""
    result = flux.copy()
    nans = np.isnan(result)
    if nans.all():
        return np.ones_like(result)
    if nans.any():
        x = np.arange(len(result))
        result[nans] = np.interp(x[nans], x[~nans], result[~nans])
    return result


def clean_lightcurve(time: np.ndarray, flux: np.ndarray,
                     sigma: float = 3.0,
                     median_kernel: int = 5,
                     savgol_window: int = 51) -> dict:
    """
    Full cleaning pipeline: NaN handling → sigma clip → median filter → smooth.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    flux : np.ndarray
        Raw flux array.
    sigma : float
        Sigma clipping threshold.
    median_kernel : int
        Median filter kernel size.
    savgol_window : int
        Savitzky-Golay window size.

    Returns
    -------
    result : dict
        Keys: 'time', 'flux_raw', 'flux_clipped', 'flux_filtered',
              'flux_smoothed', 'clip_mask', 'n_clipped'
    """
    # Step 1: Sigma clip outliers
    flux_clipped, clip_mask = sigma_clip(flux, sigma=sigma)

    # Step 2: Interpolate NaNs from clipping
    flux_interp = _interpolate_nans(flux_clipped)

    # Step 3: Median filter for cosmic rays
    flux_filtered = median_filter(flux_interp, kernel_size=median_kernel)

    # Step 4: Light smoothing (preserve transit features)
    flux_smoothed = savgol_smooth(flux_filtered, window_length=savgol_window)

    return {
        "time": time,
        "flux_raw": flux,
        "flux_clipped": flux_clipped,
        "flux_filtered": flux_filtered,
        "flux_smoothed": flux_smoothed,
        "clip_mask": clip_mask,
        "n_clipped": int((~clip_mask).sum()),
    }
