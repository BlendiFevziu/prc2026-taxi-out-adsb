"""Label-free 2026 preview of the neighbour-taxi anchor for the fallback branch.

Fits the v44 fallback baseline and nt_anchor_normal on all twelve 2025 source files and
predicts the 2026 ranking fallback rows. It reads no 2026 label (the ranking file has none)
and selects nothing from the official score. It answers two questions before any
integration:

  1. Does an all-year refit of the baseline reproduce v45's official fallback predictions?
     If not, the integration path is not understood yet, and the run stops before any
     anchor prediction is reported.
  2. Where, and by how much, would the anchor move the 2026 fallback predictions?

Output is descriptive. No bundle is written, nothing is uploaded, nothing in research_250
or the frozen release changes. Heavy: builds a twelve-month table, so run it alone.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--official", default=Path(".cache/clock_queue_release_20260912/candidate.parquet"), type=Path)
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--tolerance", type=float, default=0.05)
args = parser.parse_args()
project, output = args.project.resolve(), args.output.resolve()
official_path = args.official if args.official.is_absolute() else project / args.official
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS
import features_geometry as MG
import features_stand_rate as MSQ
from fid_gap_cv import month_of_file
import lightgbm as lgb
import numpy as np
import pandas as pd
from research_240 import features_neighbour_taxi as NT

BASE = list(H.MERKMALE_RUECK)
WITH_NT = BASE + NT.MERKMALE
KEEP = list(dict.fromkeys([f for f in BASE if f not in MSQ.MERKMALE]
                          + ["MVT_ID_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "y", "monat", "hat_aobt"]))
geometry = MG.alle()


def fallback_table(files, name, labelled):
    h = hashlib.sha256()
    for p in sorted(project.glob("*.py")) + [Path(NT.__file__).resolve()]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    for p in files:
        h.update(p.name.encode())
        h.update(str((p.stat().st_size, p.stat().st_mtime_ns)).encode())
    cached = output / f"{name}_{h.hexdigest()[:16]}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    started = time.time()
    full = H.baue([str(p) for p in files], geometry)
    mask = full.hat_aobt.eq(0).to_numpy()
    if labelled:
        mask &= full.y.notna().to_numpy()
    built = NT.baue(full, query=mask)
    small = full.loc[mask, KEEP].copy()
    for col in NT.MERKMALE:
        small[col] = built[col].to_numpy()[mask]
    del full, built
    gc.collect()
    small.to_parquet(cached, index=False)
    print("BUILT", name, len(small), "rows", round(time.time()-started, 1), "s", flush=True)
    return small


def prepare(train, valid):
    """Identical to fallback_anchor_screen.prepare."""
    train, valid = train.copy(), valid.copy()
    train["z"] = (train.y > H.SCHWELLE_EXTREM).astype(int)
    mapping = MSQ.passe_an(train)
    train = MSQ.ausser_faltung(train)
    valid = MSQ.wende_an(mapping, valid)
    for col in H.KAT:
        if col in train:
            cats = pd.unique(train[col].astype(str))
            train[col] = pd.Categorical(train[col].astype(str), categories=cats)
            valid[col] = pd.Categorical(valid[col].astype(str), categories=cats)
    return train, valid


def reg(rounds, leaves, minimum):
    return lgb.LGBMRegressor(objective="l2", n_estimators=rounds, learning_rate=.05,
                             num_leaves=leaves, min_child_samples=minimum, verbose=-1,
                             n_jobs=args.jobs, random_state=0)


paths = sorted((project / "data").glob("training_2025-*.parquet"))
assert len(paths) == 12 and {month_of_file(p) for p in paths} == set(range(1, 13))
train = fallback_table(paths, "fallback_2025_all", labelled=True).reset_index(drop=True)
rank = fallback_table([project / "data/ranking.parquet"], "fallback_2026_ranking", labelled=False).reset_index(drop=True)
tr, va = prepare(train, rank)
z0 = tr.z.eq(0).to_numpy()
y = tr.y.to_numpy(float)
versp_v = va.versp_min.to_numpy(float)*60
classifier = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=.05, num_leaves=31,
                                min_child_samples=20, verbose=-1, n_jobs=args.jobs, random_state=0)
p = classifier.fit(tr[BASE], tr.z).predict_proba(va[BASE])[:, 1]
height = (1-H.ANTEIL_HM)*versp_v + H.ANTEIL_HM*reg(300, 15, 10).fit(tr.loc[~z0, BASE], y[~z0]).predict(va[BASE])
normal = reg(400, 31, 20).fit(tr.loc[z0, BASE], y[z0]).predict(va[BASE])
q = p ** H.HOCH
baseline = np.maximum((1-q)*normal + q*height, 0)

official = pd.read_parquet(official_path).set_index("MVT_ID_mvt", verify_integrity=True)
scored = official.loc[va.MVT_ID_mvt, "TAXITIME_SEC_mvt"].to_numpy(float)
difference = np.abs(baseline-scored)
gate = {"rows": int(len(va)), "max_abs_difference_s": float(difference.max()),
        "rms_difference_s": float(np.sqrt(np.mean(difference**2))), "tolerance_s": args.tolerance}
gate["passed"] = gate["max_abs_difference_s"] <= args.tolerance
print("GATE", json.dumps(gate), flush=True)
if not gate["passed"]:
    (output / "preview_gate_failed.json").write_text(json.dumps(gate, indent=2))
    sys.exit("All-year baseline refit does not reproduce v45's official fallback predictions; stop.")

anchor_tr, anchor_va = tr.nt_anchor.to_numpy(float), va.nt_anchor.to_numpy(float)
fit_rows = z0 & np.isfinite(anchor_tr)
offset = reg(400, 31, 20).fit(tr.loc[fit_rows, WITH_NT], y[fit_rows]-anchor_tr[fit_rows])
has = np.isfinite(anchor_va)
normal_anchor = normal.copy()
normal_anchor[has] = anchor_va[has] + offset.predict(va.loc[has, WITH_NT])
anchored = np.maximum((1-q)*normal_anchor + q*height, 0)
delta = anchored-baseline

frame = va[["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "nt_anchor", "nt_n60"]].copy()
frame["ADEP_mvt"] = va.ADEP_mvt.astype(str).to_numpy()
frame["delay_s"], frame["p_extreme"] = versp_v, p
frame["v45_official"], frame["baseline_refit"], frame["anchored"], frame["delta"] = scored, baseline, anchored, delta
frame.to_parquet(output / "preview_2026_fallback.parquet", index=False)


def moved(mask):
    d = delta[mask]
    return {"rows": int(mask.sum()), "median_delta_s": float(np.median(d)) if d.size else None,
            "mean_delta_s": float(d.mean()) if d.size else None,
            "rows_moving_over_5_min": int((np.abs(d) > 300).sum()),
            "rows_moving_over_15_min": int((np.abs(d) > 900).sum()),
            "rows_moving_over_30_min": int((np.abs(d) > 1800).sum())}


disrupted = np.nan_to_num(anchor_va, nan=-1.0) >= 1800
stamps = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
schiphol_cluster = (frame.ADEP_mvt.eq("EHAM") & stamps.dt.strftime("%Y-%m-%d").between("2026-01-02", "2026-01-06")).to_numpy()
report = {"note": "label-free preview on 2026 ranking fallback rows; no official score used or selected",
          "replication_gate": gate, "all_rows": moved(np.ones(len(va), bool)),
          "disrupted_hours": moved(disrupted), "ordinary_hours": moved(~disrupted),
          "schiphol_2_to_6_january": moved(schiphol_cluster),
          "rows_without_anchor": int((~has).sum()),
          "airports": {a: moved(frame.ADEP_mvt.eq(a).to_numpy()) for a in sorted(frame.ADEP_mvt.unique())},
          "mean_abs_delta_s": float(np.abs(delta).mean()),
          "note_on_magnitude": "a delta is a prediction change, not an error change; without 2026 labels the sign "
                               "of its effect on RMSE is unknown"}
(output / "preview_2026.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ("replication_gate", "all_rows", "disrupted_hours", "ordinary_hours",
                                         "schiphol_2_to_6_january", "rows_without_anchor")}, indent=2), flush=True)
