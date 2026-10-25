"""Build v66 = the exact scored v65 with the frozen stand model switched on for Zurich July and Amsterdam July, both passing
research_440_fresh_july/PREREG_stand_july.md. NM-free. No upload.

As research_320/build_stand_candidate.py (v57): eligible rows are 2026 scored main-lane rows of Codex's context table in
the two cells whose category is observed_at_assigned_stand, owned by the base in v65 (detector rows left alone) and
sitting exactly at the comparator; the 36th input (the guarded stand detector's taxi time) comes from re-running Codex's
full_coverage_2026.extract_day with summarise() extended, and that re-extraction must reproduce his 23 features and
categories exactly. New value max(comparator + 0.5 x mean stand correction, 300).
Run from the repository root:  python3 -u -B research_440_fresh_july/build_v66.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research_360_integration"))
from research_320 import stand_experiment as S  # noqa: E402  (loads the frozen v56 modules)

MS, R = S.MS, S.R
import submission as SUB  # noqa: E402
from research_310 import full_coverage_2026 as F  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402
import harness as H  # noqa: E402

ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 256.6145
CELLS = ("LSZH/07", "EHAM/07")
C = HERE / ".cache/codex_v56"
V65 = HERE / ".cache/research_400/v65/candidate_v65.parquet"
CACHE = HERE / ".cache/research_440/stand_2026"
OUT = HERE / ".cache/research_440/v66"
REPORT = HERE / "research_440_fresh_july/results/v66_build.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rec65 = json.loads((HERE / "results/upbeat-goblin_v65.json").read_text())
    test = json.loads((HERE / "research_440_fresh_july/results/stand_july.json").read_text())
    checks = {"v65_is_scored_file": sha(V65) == rec65["file_sha256"], "test_passed": test["verdict"] == "pass",
              "cells_passed": sorted(test["passing_cells"]) == sorted(CELLS), "stand_models": S.models()["train_rows"] > 0}
    if not all(checks.values()):
        raise SystemExit(f"a precondition failed: {checks}")
    v65 = pd.read_parquet(V65)
    context = pd.read_parquet(C / "partial_release/context.parquet")
    codex = pd.read_parquet(C / "full_coverage_2026/features.parquet").set_index(ID)
    context["cell"] = context.airport + "/" + context.date.str[5:7]
    wanted = context[context.cell.isin(CELLS)]
    raw = pd.read_parquet(HERE / "data/ranking.parquet", columns=[ID, "PHASE_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt",
                                                                  "MVT_TIME_UTC_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "STAND_mvt"])
    dep = raw[raw.PHASE_mvt.eq("DEP") & raw.TAXITIME_SEC_mvt.isna() & raw.AOBT_3_flt.notna()].copy()
    dep["date"] = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    dep["airport"] = dep.ADEP_mvt.astype(str)
    dep["callsign"] = dep.FLIGHT_mvt.astype(str).str.strip().str.upper()
    dep["mvt_epoch"] = (pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True) - R.MD.EPOCH0).dt.total_seconds()
    mapping = R.AT.OB.callsign_mapping()
    maps = {ap: (R.SD.stand_table(ap), GC.xplane_table(ap)[0] if ap == "EHAM" else {}) for ap in MS.AIRPORTS}
    original = MS.summarise
    MS.summarise = lambda p, stand, takeoff: {**original(p, stand, takeoff), "detector_taxi_s": S.detector_taxi(p, stand, takeoff)}
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    try:
        with S.R.heavy_work("research_440: 2026 re-extraction for the stand model (LSZH, EHAM July)", wait=True):
            for date in sorted(wanted.date.unique()):
                cache = CACHE / f"day_{date}.parquet"
                if not cache.exists():
                    airports = sorted(wanted.loc[wanted.date.eq(date), "airport"].unique())
                    day = dep[dep.date.eq(date) & dep.airport.isin(airports)]
                    F.extract_day(day, pd.read_parquet(R.MD.ADSB / f"tracks/{date}.parquet"), mapping, maps).to_parquet(cache, index=False)
                frames.append(pd.read_parquet(cache))
    finally:
        MS.summarise = original
    mine = pd.concat(frames, ignore_index=True).set_index(ID)
    theirs = codex.loc[mine.index]
    cols = ["category"] + list(MS.FEATURES)
    checks["extraction_equals_codex"] = bool(all(
        np.array_equal(mine[c].to_numpy(), theirs[c].to_numpy()) if c == "category"
        else np.array_equal(mine[c].to_numpy(float), theirs[c].to_numpy(float), equal_nan=True) for c in cols))
    if not checks["extraction_equals_codex"]:
        raise SystemExit("the re-extraction does not reproduce Codex's 2026 features")
    elig = wanted[wanted.category.eq("observed_at_assigned_stand")].copy()
    own = H.ownership_v62().set_index(ID).owner
    checks["stand_rows_by_owner"] = own.loc[elig[ID]].value_counts().to_dict()
    elig = elig[own.loc[elig[ID]].eq("base").to_numpy()].copy()
    at65 = v65.set_index(ID).loc[elig[ID], T].to_numpy(float)
    same = elig.comparator.to_numpy(float) == at65
    checks["base_owned_rows_not_at_comparator"] = int((~same).sum())
    elig = elig[same].reset_index(drop=True)
    elig["detector_taxi_s"] = mine.loc[elig[ID], "detector_taxi_s"].to_numpy(float)
    checks["rows_by_cell"] = elig.cell.value_counts().to_dict()
    x = elig[S.COLS]
    corr = np.mean([R.joblib.load(S.CACHE / "models" / f"real_{s}.pkl").predict(x) for s in S.SEEDS], axis=0)
    pred = np.maximum(elig.comparator.to_numpy(float) + S.SHRINK * corr, 300.0)
    rev = x.iloc[::-1]
    replay = np.maximum(elig.comparator.to_numpy(float)[::-1] + S.SHRINK * np.mean(
        [R.joblib.load(S.CACHE / "models" / f"real_{s}.pkl").predict(rev) for s in S.SEEDS], axis=0), 300.0)[::-1]
    checks["reverse_order_replay_equal"] = bool(np.array_equal(pred, replay))
    checks["shrink"] = float(S.SHRINK)
    cand = v65.copy()
    update = pd.Series(pred, index=elig[ID].to_numpy())
    hit = cand[ID].isin(update.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(update).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v65.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v65.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v65=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int((cand[T].to_numpy() != v65[T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    ok = not errors and untouched and checks["all_finite"] and checks["reverse_order_replay_equal"]
    rows26 = {c: int(v) for c, v in checks["rows_by_cell"].items()}
    mse = sum(rows26[c] * test["cells"][c]["gain_mse"] for c in CELLS) / N
    at = lambda k: round(REF - (REF - np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v65", "score": rec65["score"], "sha256": rec65["file_sha256"]},
              "prediction_fixed_before_upload": {"assumptions": "per-cell gains on the ten fresh July days (weight 0.5 included) times the "
                                                                "eligible 2026 rows; central 0.8, worst 0.3, best 1.2 of full transfer",
                                                 "mse_per_scored_row_full": mse, "central": at(0.8), "worst": at(0.3), "best": at(1.2)}}
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v66.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
