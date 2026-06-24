"""
Detrending Module for ExoplanetAI

Removes long-term systematic trends from light curves to
flatten them around normalized flux = 1.0.

Techniques:
- Polynomial detrending
- Sliding window median normalization
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def polynomial_detrend(time: np.ndarray, flux: np.ndarray,
                       order: int = 2) -> np.ndarray:
    """
    Remove long-term trend using polynomial fit subtraction.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    flux : np.ndarray
        Flux array (should already be cleaned of major outliers).
    order : int
        Polynomial order (1=linear, 2=quadratic, etc.).

    Returns
    -------
    detrended : np.ndarray
        Detrended flux centered around 1.0.
    """
    mask = np.isfinite(flux)
    if mask.sum() < order + 1:
        return flux.copy()

    coeffs = np.polyfit(time[mask], flux[mask], order)
    trend = np.polyval(coeffs, time)

    # Detrend and re-centre at 1.0
    detrended = flux / trend
    return detrended


def sliding_window_detrend(flux: np.ndarray,
                           window_size: int = 201) -> np.ndarray:
    """
    Normalize flux using a sliding window median.

    Divides flux by a running median to remove slow variations
    while preserving transit-timescale features.

    Parameters
    ----------
    flux : np.ndarray
        Input flux array.
    window_size : int
        Window size for running median (should be >> transit duration).

    Returns
    -------
    detrended : np.ndarray
        Flux divided by the running median, centred at 1.0.
    """
    if window_size % 2 == 0:
        window_size += 1

    # Use uniform filter as fast approximation of running median
    # (true running median is O(n*w); this is O(n))
    trend = uniform_filter1d(flux.astype(float), size=window_size, mode='nearest')

    # Avoid division by zero
    trend[trend == 0] = 1.0

    return flux / trend


def detrend_lightcurve(time: np.ndarray, flux: np.ndarray,
                       method: str = "polynomial",
                       poly_order: int = 2,
                       window_size: int = 201) -> np.ndarray:
    """
    Apply detrending to a light curve.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    flux : np.ndarray
        Flux array.
    method : str
        'polynomial' or 'sliding_window'.
    poly_order : int
        Polynomial order (if method='polynomial').
    window_size : int
        Window size (if method='sliding_window').

    Returns
    -------
    detrended : np.ndarray
        Detrended flux array.
    """
    if method == "polynomial":
        return polynomial_detrend(time, flux, order=poly_order)
    elif method == "sliding_window":
        return sliding_window_detrend(flux, window_size=window_size)
    else:
        raise ValueError(f"Unknown detrending method: {method}")
