"""Production-strength test of detector v2 at Munich: four fresh January 2025 dates against the held-out v45 ensemble.

v2 passed at Munich in December 2025 against one LightGBM (PREREG_stand_detector.md, addendum C), a comparator
about 7 s weaker than production. January 2025 is the one month where Munich had ADS-B ground coverage AND
production-strength held-out predictions exist: those of the full v45 ensemble, trained without January and July
2025 (.cache/sequence_release_v47/validate/ensemble_baseline.parquet, column `baseline`, 196.95 s on January's
152,271 main-lane rows). Its SHA-256 is checked against ensemble_baseline.json before use.

Rules, dates and decision: PREREG_stand_detector.md addendum D, committed and pushed before three of the four dates
were downloaded. Decision rules V1 to V4 of section 4, unchanged. This script only passes the v2 detector, these
dates, this comparator and new output names to munich_validate.main.

Run from the repository root:  python3 -u -B research_290/munich_validate_jan2025.py
"""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_290 import munich_validate as MV  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402

DATES = ("2025-01-08", "2025-01-16", "2025-01-24", "2025-01-26")
OUTPUT = ROOT / "research_290/results/munich_validate_jan2025.json"
VALIDATE = ROOT / ".cache/sequence_release_v47/validate"


def ensemble():
    path, info = VALIDATE / "ensemble_baseline.parquet", json.loads((VALIDATE / "ensemble_baseline.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["baseline_sha256"]:
        raise SystemExit("ensemble_baseline.parquet does not match its recorded SHA-256")
    frame = pd.read_parquet(path, columns=["MVT_ID_mvt", "y", "ersatz_s", "baseline"])
    return frame.rename(columns={"baseline": "pred"})


def main():
    MV.main(dates=DATES, detect=SD.detect_onset_v2, output=OUTPUT, rows_file="validation_rows_jan2025.parquet",
            preregistration="research_290/PREREG_stand_detector.md addendum D", comparator=ensemble(),
            comparator_name="v45 full ensemble, held-out January 2025 (ensemble_baseline.parquet)")


if __name__ == "__main__":
    main()
