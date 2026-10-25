"""Label-free: the 12 ADS-B and 6 stand columns for the January and July 2026 main-lane rows (ranking file), for the rebuild's
2026 build. No prediction is made here; the corrector is applied only after pilot 7 (PREREG_transfer.md) passes.
Run from the repository root:  python3 -u -B research_730_adsb_learner/build_2026_table.py
"""
import glob
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from research_730_adsb_learner import features_adsb as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/research_730"
TRACKS = ROOT.parent / "adsb-lol/tracks"
OUT = CACHE / "table_2026_1_7_stand.parquet"


def main():
    release = pd.read_parquet(ROOT / ".cache/clock_queue_release_20260912/data/ranking.parquet",
                              columns=["MVT_ID_mvt", "hat_aobt", "ersatz_s", "stunde", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    release = release[release.hat_aobt.eq(1)].reset_index(drop=True)
    raw = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["MVT_ID_mvt", "CALLSIGN_flt", "STAND_mvt"]).drop_duplicates("MVT_ID_mvt")
    frame = release.merge(raw, on="MVT_ID_mvt", how="left")
    files = sorted(glob.glob(str(TRACKS / "2026-01-??.parquet")) + glob.glob(str(TRACKS / "2026-07-??.parquet")))
    print(f"{len(frame):,} main-lane rows, {len(files)} track days", flush=True)
    tracks = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = F.compute(frame, tracks, stands=True)
    assert np.array_equal(out.index.to_numpy(), frame.MVT_ID_mvt.to_numpy())
    for c in F.MERKMALE + F.STAND_MERKMALE:
        frame[c] = out[c].to_numpy()
    frame.to_parquet(OUT, index=False)
    print(f"matched {frame.ad_matched.mean():.3f}, stand ok {frame.sd_status.eq(1).mean():.3f}; by airport matched:",
          frame.groupby("ADEP_mvt").ad_matched.mean().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main()
