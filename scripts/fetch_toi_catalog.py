"""
Fetch TESS Objects of Interest (TOI) Catalog

This script downloads the official TOI catalog from the NASA Exoplanet Archive.
It parses the catalog into labeled categories for supervised machine learning:
- Confirmed Planets (CP) -> 'transit'
- False Positives (FP) -> 'eclipsing_binary' or 'blend'
- Others -> 'noise' or 'variable_star'
"""

import os
import pandas as pd
from pathlib import Path

TOI_API_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv"

def fetch_toi_catalog(output_path: str) -> pd.DataFrame:
    """Download the TOI catalog and save it locally."""
    print(f"📡 Downloading TOI Catalog from NASA Exoplanet Archive...")
    try:
        df = pd.read_csv(TOI_API_URL)
        df.to_csv(output_path, index=False)
        print(f"✅ TOI Catalog saved to {output_path} ({len(df)} targets found)")
        return df
    except Exception as e:
        print(f"❌ Failed to download TOI Catalog: {e}")
        return pd.DataFrame()


def categorize_targets(df: pd.DataFrame) -> dict:
    """Categorize targets by TFOPWG Disposition."""
    if df.empty or "tfopwg_disp" not in df.columns:
        print("⚠️ Missing TFOPWG Disposition column.")
        return {}

    # NASA Exoplanet Archive TOI dispositions:
    # CP: Confirmed Planet
    # KP: Known Planet
    # FP: False Positive (often eclipsing binaries)
    # PC: Planet Candidate (unconfirmed)
    
    transit_targets = df[df["tfopwg_disp"].isin(["CP", "KP"])]
    fp_targets = df[df["tfopwg_disp"] == "FP"]
    candidate_targets = df[df["tfopwg_disp"] == "PC"]

    print("\n📊 TOI Catalog Breakdown:")
    print(f"   Confirmed/Known Planets (Transit): {len(transit_targets)}")
    print(f"   False Positives (Eclipsing Binaries/Blends): {len(fp_targets)}")
    print(f"   Unconfirmed Candidates: {len(candidate_targets)}")

    return {
        "transit": transit_targets["tid"].dropna().unique().tolist(),
        "eclipsing_binary": fp_targets["tid"].dropna().unique().tolist(),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "tess_labeled"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    cat_path = data_dir / "toi_catalog.csv"
    df = fetch_toi_catalog(str(cat_path))
    if not df.empty:
        categorize_targets(df)
