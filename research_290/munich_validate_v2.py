"""Validation of detector v2 (stand_detector.detect_onset_v2) on four FRESH dates: Munich, 5, 8, 11, 14 December 2025.

v2 is v1 plus one rule, added after v1 failed its validation (PREREG addendum A). The rule, these dates and the
decision are fixed in PREREG addendum B, committed and pushed before any data of these dates was downloaded.
Everything else is v1's: decision rules V1 to V4 of section 4 with their thresholds, the comparator predictions
(.cache/munich_stand_detector/december_model.parquet), the bootstrap seed and the 5 % trimming. This script only
passes a different detector, dates and output names to munich_validate.main.

Run from the repository root:  python3 -u -B research_290/munich_validate_v2.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_290 import munich_validate as MV  # noqa: E402
from research_290 import stand_detector as SD  # noqa: E402

DATES = ("2025-12-05", "2025-12-08", "2025-12-11", "2025-12-14")
OUTPUT = ROOT / "research_290/results/munich_validate_v2.json"
PREREG = "research_290/PREREG_stand_detector.md"


# The commit that registered addendum B, fixed here. It used to be read from git at run time, which made a later
# rerun name whichever commit had last touched the file (addendum C moved it to 791a5cd). The wording is that of
# the recorded run, so a rerun reproduces results/munich_validate_v2.json byte for byte.
REGISTERED = "ffccea1"


def main():
    MV.main(dates=DATES, detect=SD.detect_onset_v2, output=OUTPUT, rows_file="validation_rows_v2.parquet",
            preregistration=f"{PREREG} addendum B (last commit touching it: {REGISTERED})")


if __name__ == "__main__":
    main()
