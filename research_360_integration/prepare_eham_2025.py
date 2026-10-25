"""Amsterdam January features for the 2025 evaluation days that were never extracted (6, 13 and 20 January, the old
NM days). No fit, no label use: Codex's eham_recovery.extract from the read-only checkout of 8d98953, which asserts
its own replay of the category and 23 v56 summaries against the reference rows (research_320 extraction tables)
before adding the seven sampling-aware features. It takes the shared lock itself.
Run from the repository root:  python3 -u -B research_360_integration/prepare_eham_2025.py
"""
from pathlib import Path
import subprocess
import sys

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
EH = HERE.parent / "skenderbeu-eham-readonly"
OUT = HERE / ".cache/research_360/eham_features"
DAYS = ("2025-01-06", "2025-01-13", "2025-01-20")
ID = "MVT_ID_mvt"


def main():
    head = subprocess.run(["git", "-C", str(EH), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    if not head.startswith("8d98953"):
        raise SystemExit("the EHAM checkout is not 8d98953")
    sys.path.insert(0, str(EH))
    import main_model  # noqa: F401  (torch before tree libraries)
    from research_310 import eham_recovery as ER
    stand = HERE / ".cache/research_320/stand"
    ref = pd.concat([pd.read_parquet(stand / "development.parquet"), pd.read_parquet(stand / "confirm.parquet")], ignore_index=True)
    ref = ref[ref.date.isin(DAYS)].drop_duplicates(ID)
    OUT.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(OUT / "reference.parquet", index=False)
    ER.reference = lambda date: OUT / "reference.parquet"
    ER.OUT = OUT
    for day in DAYS:
        f = ER.extract(day)
        print(day, len(f), f.category.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
