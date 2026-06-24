"""
TESS Light Curve Downloader

Downloads real TESS light curves using the lightkurve library.
Supports single-target and batch downloads.
"""

import os
from pathlib import Path

try:
    import lightkurve as lk
    HAS_LIGHTKURVE = True
except ImportError:
    HAS_LIGHTKURVE = False

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Known exoplanet targets for validation
# ---------------------------------------------------------------------------
KNOWN_TARGETS = {
    "WASP-18b":    "TIC 25155310",
    "WASP-121b":   "TIC 22529346",
    "TOI-700d":    "TIC 150428135",
    "HD 209458b":  "HD 209458",
    "HAT-P-7b":    "TIC 424865156",
    "KELT-9b":     "TIC 228732031",
    "55 Cnc e":    "TIC 332064670",
    "GJ 357 b":    "TIC 413248763",
}


def download_single(target: str, output_dir: str, mission: str = "TESS",
                     sector: int = None) -> str | None:
    """
    Download a single TESS light curve.

    Parameters
    ----------
    target : str
        Target name or TIC ID.
    output_dir : str
        Directory to save the CSV.
    mission : str
        Mission name (default: TESS).
    sector : int, optional
        Specific sector to download.

    Returns
    -------
    filepath : str or None
        Path to the saved CSV, or None on failure.
    """
    if not HAS_LIGHTKURVE:
        print("❌ lightkurve not installed. Run: pip install lightkurve")
        return None

    try:
        search_kwargs = {"target": target, "mission": mission}
        if sector is not None:
            search_kwargs["sector"] = sector

        search_result = lk.search_lightcurve(**search_kwargs)

        if len(search_result) == 0:
            print(f"⚠️  No data found for {target}")
            return None

        # Download first result
        lc = search_result[0].download()
        lc = lc.remove_nans().normalize()

        # Convert to DataFrame
        df = pd.DataFrame({
            "time": lc.time.value,
            "flux": lc.flux.value,
            "flux_err": lc.flux_err.value if lc.flux_err is not None else np.zeros(len(lc.time)),
        })

        os.makedirs(output_dir, exist_ok=True)
        safe_name = target.replace(" ", "_").replace("/", "_")
        filepath = os.path.join(output_dir, f"{safe_name}.csv")
        df.to_csv(filepath, index=False)
        print(f"✅ Downloaded {target} → {filepath}  ({len(df)} points)")
        return filepath

    except Exception as e:
        print(f"❌ Failed to download {target}: {e}")
        return None


def download_known_exoplanets(output_dir: str) -> list[str]:
    """Download light curves for all known exoplanet targets."""
    paths = []
    for name, tic_id in KNOWN_TARGETS.items():
        print(f"📡 Downloading {name} ({tic_id})...")
        path = download_single(tic_id, output_dir)
        if path:
            paths.append(path)
    return paths


def download_sector(sector: int, output_dir: str, max_targets: int = 100) -> list[str]:
    """
    Download light curves from a TESS sector.

    Parameters
    ----------
    sector : int
        TESS sector number (1-70+).
    output_dir : str
        Output directory.
    max_targets : int
        Max number of targets to download.

    Returns
    -------
    paths : list[str]
        List of saved CSV file paths.
    """
    if not HAS_LIGHTKURVE:
        print("❌ lightkurve not installed.")
        return []

    try:
        search_result = lk.search_lightcurve(mission="TESS", sector=sector)
        print(f"📡 Found {len(search_result)} targets in TESS Sector {sector}")

        paths = []
        for i, item in enumerate(search_result[:max_targets]):
            try:
                lc = item.download()
                lc = lc.remove_nans().normalize()

                df = pd.DataFrame({
                    "time": lc.time.value,
                    "flux": lc.flux.value,
                    "flux_err": lc.flux_err.value if lc.flux_err is not None else np.zeros(len(lc.time)),
                })

                fname = f"sector{sector}_{i:04d}.csv"
                fpath = os.path.join(output_dir, fname)
                df.to_csv(fpath, index=False)
                paths.append(fpath)

                if (i + 1) % 10 == 0:
                    print(f"   Downloaded {i + 1}/{min(len(search_result), max_targets)}")
            except Exception as e:
                print(f"   ⚠️  Skipped target {i}: {e}")
                continue

        print(f"✅ Downloaded {len(paths)} light curves from Sector {sector}")
        return paths

    except Exception as e:
        print(f"❌ Sector download failed: {e}")
        return []


def download_labeled_tess(output_base_dir: str, n_per_class: int = 10):
    """
    Download a balanced subset of labeled TESS light curves using TOI catalog.
    """
    from scripts.fetch_toi_catalog import fetch_toi_catalog, categorize_targets
    cat_path = os.path.join(output_base_dir, "toi_catalog.csv")
    df = fetch_toi_catalog(cat_path)
    if df.empty:
        return
        
    categories = categorize_targets(df)
    
    for label, tic_ids in categories.items():
        class_dir = os.path.join(output_base_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        print(f"\n📡 Downloading {label} targets...")
        
        success_count = 0
        for tid in tic_ids:
            if success_count >= n_per_class:
                break
            target_name = f"TIC {tid}"
            print(f"  Fetching {target_name} ({success_count+1}/{n_per_class})...")
            # We don't specify sector to get the first available sector
            path = download_single(target_name, class_dir)
            if path:
                success_count += 1

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parent.parent
    tess_dir = project_root / "data" / "tess_labeled"
    
    n_per_class = 10
    if len(sys.argv) > 1:
        n_per_class = int(sys.argv[1])
        
    download_labeled_tess(str(tess_dir), n_per_class=n_per_class)
