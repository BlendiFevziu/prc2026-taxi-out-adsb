"""Assemble the single final submission (upbeat-goblin_final.parquet) for the PRC Data Challenge 2026 final phase.

Rows and sources
  January and July 2026 (the 344,841 departures that were already in ranking.parquet):
      the scored v80 submission (NM-free, 245.046 on the leaderboard) plus the known shift of MVT_TIME_UTC_mvt between
      ranking.parquet and final_ranking.parquet. The label is MVT_TIME minus the block time, the export dithered
      MVT_TIME by up to +-12 s (92 % of rows) and left every _flt column and SCHED_TIME unchanged, so the prediction
      moves with it. The probe-derived Rome values and group shifts of research_700_probe are NOT used.
  February and June 2026 (325,949 new departures):
      research_800_final/four_month_base.py: the frozen NM-free base (v45 all-year ensemble with the five sequence
      nets, anchored fallback, Rome copy gate and the v69/v70 Rome rules), plus any ADS-B layer listed in
      --layers (parquet files with MVT_ID_mvt, TAXITIME_SEC_mvt) that passed its own check.

Checks before writing (all abort on failure): ids and order equal final_submitting.parquet, dtype, finite, non-negative,
every row from exactly one source, Jan/Jul rows equal v80 + shift within the floor rule.
Nothing is uploaded here.

Run from the repository root:  python3 -u -B research_800_final/assemble_final.py [--layers a.parquet b.parquet]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
OUT = ROOT / ".cache/research_800/final"
V80 = ROOT / ".cache/research_640/v80/candidate_v80.parquet"
BASE = ROOT / ".cache/research_800/base_final_predictions.parquet"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(layers):
    template = pd.read_parquet(ROOT / "data/final_submitting.parquet")
    final = pd.read_parquet(ROOT / "data/final_ranking.parquet", columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    old = pd.read_parquet(ROOT / "data/ranking.parquet", columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    v80 = pd.read_parquet(V80)
    base = pd.read_parquet(BASE)
    assert base[ID].is_unique and v80[ID].is_unique
    dep_new = final[final.PHASE_mvt.eq("DEP")].set_index(ID).MVT_TIME_UTC_mvt
    dep_old = old[old.PHASE_mvt.eq("DEP")].set_index(ID).MVT_TIME_UTC_mvt

    # Jan/Jul: v80 plus the MVT_TIME shift
    v80s = v80.set_index(ID)[T]
    shift = (pd.to_datetime(dep_new.loc[v80s.index], utc=True) - pd.to_datetime(dep_old.loc[v80s.index], utc=True)).dt.total_seconds()
    floor = np.minimum(v80s.to_numpy(float), 300.0)             # v80's 300 s floor stays where it was active
    jan_jul = pd.Series(np.maximum(v80s.to_numpy(float) + shift.to_numpy(float), floor), index=v80s.index)

    # Feb/Jun: base plus validated layers
    feb_jun_ids = template[ID][~template[ID].isin(jan_jul.index)].to_numpy()
    values = base.set_index(ID)[T].loc[feb_jun_ids].copy()
    sources = {"v80_plus_shift": int(len(jan_jul)), "base": int(len(values))}
    for layer in layers:
        extra = pd.read_parquet(layer).set_index(ID)[T]
        hit = extra.index.intersection(values.index)
        values.loc[hit] = extra.loc[hit].to_numpy(float)
        sources[Path(layer).name] = int(len(hit))

    out = template[[ID]].copy()
    combined = pd.concat([jan_jul, values])
    assert combined.index.is_unique
    out[T] = combined.loc[out[ID].to_numpy()].to_numpy(float)
    out[T] = out[T].astype(template[T].dtype)

    errors = []
    if len(out) != len(template):
        errors.append("row count")
    if not out[ID].equals(template[ID]):
        errors.append("ids or order differ from final_submitting.parquet")
    if str(out[T].dtype) != str(template[T].dtype):
        errors.append("dtype")
    if not np.isfinite(out[T].to_numpy(float)).all():
        errors.append("non-finite values")
    if (out[T] < 0).any():
        errors.append("negative values")
    if list(out.columns) != list(template.columns):
        errors.append("columns")
    if errors:
        raise SystemExit(f"CHECK FAILED: {errors}")

    month = pd.to_datetime(dep_new.loc[out[ID].to_numpy()], utc=True).dt.strftime("%Y-%m").to_numpy()
    summary = pd.DataFrame({"month": month, "p": out[T].to_numpy(float)}).groupby("month").p.agg(["size", "mean", "median"])
    print(summary.round(1).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "upbeat-goblin_final.parquet"
    out.to_parquet(target, index=False)
    report = {"file": str(target), "sha256": sha(target), "rows": int(len(out)), "sources": sources,
              "layers": [str(Path(p).name) for p in layers], "v80_sha256": sha(V80), "base_sha256": sha(BASE),
              "shift_mvt_s": {"mean": float(shift.mean()), "sd": float(shift.std()), "max_abs": float(shift.abs().max())},
              "by_month": json.loads(summary.round(3).to_json(orient="index"))}
    (OUT / "assemble_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("sha256", "rows", "sources", "layers")}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="*", default=[])
    main(parser.parse_args().layers)
