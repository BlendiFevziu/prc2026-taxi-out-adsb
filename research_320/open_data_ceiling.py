"""LABEL-FREE ceiling: how much could open ADS-B still add to v56 (261.2616)? Counts, not a model; nothing is fitted.

Pools of 2026 scored main-lane rows with open ADS-B evidence that no deployed component uses yet (categories from
Codex's full_coverage_2026.json at 9884150, Codex's own frozen classifier), each given an optimistic per-row MSE gain:
  clean_stand   observed_at_assigned_stand at EDDF, EGLL, LEBL, LSZH (v56 corrects only partial tracks; the stand
                detector is validated only at EDDM and EHAM). Upper bound: detector-grade gain.
  withheld      partial tracks in v56's withheld cells (EDDF July, EGLL July closed by the coverage audit; EHAM
                January is Codex's). Upper bound: v56's best cell gain.
The rows without any ground evidence (the four airports LTFM, LEMD, LFPG, LIRF above all) get nothing from ADS-B.
Run from the repository root:  python3 -u -B research_320/open_data_ceiling.py
"""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "research_320/results/open_data_ceiling.json"
V56, N = 261.2616, 344841


def seconds(mse_per_scored_row):
    return float(V56 - np.sqrt(V56 ** 2 - mse_per_scored_row))


def main():
    cov = json.loads(subprocess.run(["git", "show", "9884150:research_310/results/full_coverage_2026.json"], cwd=HERE,
                                    capture_output=True, text=True, check=True).stdout)["cells"]
    clean = {c: v["category_counts"].get("observed_at_assigned_stand", 0) for c, v in cov.items() if c[:4] in ("EDDF", "EGLL", "LEBL", "LSZH")}
    withheld = {"EDDF/07": 10824, "EGLL/07": 8136, "EHAM/01": 8968}      # after NM/detector protection (Codex's audit)
    r = pd.read_parquet(HERE / "data/ranking.parquet", columns=["PHASE_mvt", "TAXITIME_SEC_mvt", "ADEP_mvt", "AOBT_3_flt"])
    r = r[r.PHASE_mvt.eq("DEP") & r.TAXITIME_SEC_mvt.isna() & r.AOBT_3_flt.notna()]
    no_ground_airports = int(r.ADEP_mvt.astype(str).isin(["LTFM", "LEMD", "LFPG", "LIRF"]).sum())
    need = {s: float(V56 ** 2 - (V56 - s) ** 2) for s in (5, 8)}
    pools = {"clean_stand": sum(clean.values()), "withheld_partial": sum(withheld.values())}
    scenarios = {}
    for name, rows, gains in (("clean_stand", pools["clean_stand"], (5000, 10000, 20000)),
                              ("withheld_partial", pools["withheld_partial"], (2800, 6400))):
        scenarios[name] = {str(g): round(seconds(rows * g / N), 3) for g in gains}
    both = seconds((pools["clean_stand"] * 20000 + pools["withheld_partial"] * 6400) / N)
    report = {"v56": V56, "mse_per_scored_row_needed": need, "clean_stand_rows": clean, "withheld_rows": withheld,
              "pools": pools, "main_rows_at_LTFM_LEMD_LFPG_LIRF": no_ground_airports, "main_rows": int(len(r)),
              "seconds_by_gain_per_row": scenarios, "upper_bound_everything_s": round(both, 3)}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
