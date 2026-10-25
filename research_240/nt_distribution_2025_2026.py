"""2025 against 2026 distribution of the neighbour-taxi columns on fallback rows.

Recommended before training by another team on the competition Discord (another participant,
12.09.2026), who lost 3 s to a count feature whose scale depended on the months of history
a row had. The comparison is like for like: 2025 uses the screen's cached TWO-month
held-out tables, the same shape as the two-month 2026 ranking file, and 2026 is built with
the same builders. The ranking file carries no departure label, so none is read.
"""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--screen", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
project, screen, output = args.project.resolve(), args.screen.resolve(), args.output.resolve()
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import features_geometry as MG
import numpy as np
import pandas as pd
from research_240 import features_neighbour_taxi as NT

COLUMNS = ["ADEP_mvt", "MVT_TIME_UTC_mvt"] + NT.MERKMALE
sides = [s for s in sorted(screen.glob("fallback_nt_*.json"))
         if len(json.loads(s.read_text())["source_months"]) == 2]
months = sorted(m for s in sides for m in json.loads(s.read_text())["source_months"])
if months != list(range(1, 13)):
    raise SystemExit(f"Two-month 2025 tables do not cover every month once: {months}")
y2025 = pd.concat([pd.read_parquet(s.with_suffix(".parquet"), columns=COLUMNS) for s in sides],
                  ignore_index=True)

full = H.baue([str(project / "data/ranking.parquet")], MG.alle())
mask = full.hat_aobt.eq(0).to_numpy()
built = NT.baue(full, query=mask)
y2026 = full.loc[mask, ["ADEP_mvt", "MVT_TIME_UTC_mvt"]].copy()
for col in NT.MERKMALE:
    y2026[col] = built[col].to_numpy()[mask]
for frame in (y2025, y2026):
    frame["ADEP_mvt"] = frame.ADEP_mvt.astype(str)


def describe(values):
    finite = values[np.isfinite(values)]
    q = np.quantile(finite, [.1, .5, .9, .99]) if finite.size else [np.nan]*4
    return {"rows": int(values.size), "missing_share": float(1-finite.size/max(values.size, 1)),
            "p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2]), "p99": float(q[3]),
            "share_ge_1800": float((finite >= 1800).mean()) if finite.size else None}


def psi(old, new):
    old, new = old[np.isfinite(old)], new[np.isfinite(new)]
    if old.size < 50 or new.size < 50:
        return None
    edges = np.unique(np.quantile(old, np.linspace(0, 1, 11)))
    if edges.size < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(old, edges)[0]/old.size
    b = np.histogram(new, edges)[0]/new.size
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return float(((b-a)*np.log(b/a)).sum())


report = {"note": "2025 two-month held-out fallback tables vs 2026 ranking fallback rows; "
                  "PSI above 0.25 is conventionally a major shift",
          "rows_2025": int(len(y2025)), "rows_2026": int(len(y2026)), "overall": {}, "airports": {}}
for col in NT.MERKMALE:
    old, new = y2025[col].to_numpy(float), y2026[col].to_numpy(float)
    report["overall"][col] = {"2025": describe(old), "2026": describe(new), "psi": psi(old, new)}
for airport in sorted(set(y2025.ADEP_mvt) | set(y2026.ADEP_mvt)):
    a = y2025[y2025.ADEP_mvt == airport]
    b = y2026[y2026.ADEP_mvt == airport]
    report["airports"][airport] = {
        col: {"2025": describe(a[col].to_numpy(float)), "2026": describe(b[col].to_numpy(float)),
              "psi": psi(a[col].to_numpy(float), b[col].to_numpy(float))}
        for col in ("nt_anchor", "nt_n60")}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2))
print(f"{'column':12s}{'2025 p50':>10}{'2026 p50':>10}{'2025 p90':>10}{'2026 p90':>10}{'2025 miss':>11}{'2026 miss':>11}{'PSI':>8}")
for col, r in report["overall"].items():
    print(f"{col:12s}{r['2025']['p50']:>10.0f}{r['2026']['p50']:>10.0f}{r['2025']['p90']:>10.0f}{r['2026']['p90']:>10.0f}"
          f"{r['2025']['missing_share']*100:>10.1f}%{r['2026']['missing_share']*100:>10.1f}%"
          f"{(r['psi'] if r['psi'] is not None else float('nan')):>8.3f}")
print("\nnt_anchor by airport: share of fallback rows in disrupted hours (>= 30 min), and PSI")
for airport, r in report["airports"].items():
    x = r["nt_anchor"]
    s25, s26 = x["2025"]["share_ge_1800"], x["2026"]["share_ge_1800"]
    print(f"  {airport}  2025 {x['2025']['rows']:>6,} rows {(s25 or 0)*100:5.1f}%   "
          f"2026 {x['2026']['rows']:>6,} rows {(s26 or 0)*100:5.1f}%   PSI {x['psi'] if x['psi'] is not None else float('nan'):.3f}")
