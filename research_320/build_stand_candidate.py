"""Build v57 = the exact scored v56 plus the stand overlay (research_320/PREREG_stand.md, both stages passed). No upload.

Inputs, each hash-checked: the scored v56 file (SHA-256 1767cc02...), Codex's 2026 context table (his recorded
context_sha256) and 2026 feature table (his recorded features_sha256), copied read-only from his cache with the
owner's permission to .cache/codex_v56/; the frozen stand models (research_320 manifest); the adsb.lol tracks.

Eligible rows: 2026 scored main-lane rows in Codex's context table whose category is observed_at_assigned_stand, in
the seven pre-registered cells (EDDF Jan/Jul, EGLL Jan, LEBL Jan/Jul, LSZH Jan/Jul), NOT on days 6, 13 or 20 (every
NM-covered row lies on those days, so the whole days are protected without reading any NM file), and whose context
comparator equals v56's value exactly (so no row another component changed is touched).
The 36th column (the guarded stand detector) needs the matched points, so Codex's own full_coverage_2026.extract_day
is re-run for those days and airports with summarise() extended; his 23 features and categories must come out equal
to his table. Safety gate added before any result is known (it can only remove cells): v56's own label-free drift
rule per cell on the 24 track columns, 2025 training stand rows against the 2026 eligible rows.
Run from the repository root:  python3 -u -B research_320/build_stand_candidate.py
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import stand_experiment as S  # noqa: E402  (loads the frozen v56 modules)

MS, R = S.MS, S.R
import submission as SUB  # noqa: E402  (the frozen checkout's copy)
from research_310 import full_coverage_2026 as F  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402

C = HERE / ".cache/codex_v56"
V56_FILE = C / "partial_release_v55/candidate.parquet"
V56_SHA = "1767cc0257ddebf6350f1349d3c5487303ffa2df64f4ef0f9852db0b63e7c04c"
OUT = HERE / ".cache/research_320/stand_candidate"
REPORT = HERE / "research_320/results/stand_candidate.json"
ID, NM_DAYS = "MVT_ID_mvt", ("06", "13", "20")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded(name):
    return json.loads(subprocess.run(["git", "show", f"9884150:research_310/results/{name}"], cwd=HERE,
                                     capture_output=True, text=True, check=True).stdout)


def drift(ref, cur):
    """v56's release gate (RELEASE_PROTOCOL.md, point 2) on one cell: which of the 24 track columns fail."""
    flags = []
    for col in S.TRACK:
        a, b = ref[col].astype(float), cur[col].astype(float)
        sd = a.std()
        miss = abs(b.isna().mean() - a.isna().mean())
        if a.notna().any() and b.notna().any():
            shift = abs(b.mean() - a.mean()) / sd if sd > 0 else (0.0 if b.nunique() <= 1 and b.mean() == a.mean() else np.inf)
        else:
            shift = 0.0
        if shift > 1.0 or miss > 0.20:
            flags.append({"feature": col, "mean_shift_sd": float(shift), "missingness_change": float(miss)})
    return flags


def main():
    checks = {"v56_sha": sha(V56_FILE) == V56_SHA,
              "context_sha": sha(C / "partial_release/context.parquet") == recorded("partial_candidate.json")["context_sha256"],
              "features_sha": sha(C / "full_coverage_2026/features.parquet") == recorded("full_coverage_2026.json")["features_sha256"],
              "stand_models": S.models()["train_rows"] > 0}
    for name in ("development", "confirm"):
        checks[f"{name}_passed"] = json.loads((HERE / f"research_320/results/stand_{name}.json").read_text())["verdict"] == "pass"
    if not all(checks.values()):
        raise SystemExit(f"a precondition failed: {checks}")
    v56 = pd.read_parquet(V56_FILE)
    context = pd.read_parquet(C / "partial_release/context.parquet")
    codex = pd.read_parquet(C / "full_coverage_2026/features.parquet").set_index(ID)
    context["cell"] = context.airport + "/" + context.date.str[5:7]
    wanted = context[context.cell.isin(S.CELLS) & ~context.date.str[8:10].isin(NM_DAYS)]
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
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    try:
        for date in sorted(wanted.date.unique()):
            cache = OUT / f"day_{date}.parquet"
            if cache.exists():
                frames.append(pd.read_parquet(cache))
                continue
            airports = sorted(wanted.loc[wanted.date.eq(date), "airport"].unique())
            day = dep[dep.date.eq(date) & dep.airport.isin(airports)]
            f = F.extract_day(day, pd.read_parquet(R.MD.ADSB / f"tracks/{date}.parquet"), mapping, maps)
            f.to_parquet(cache, index=False)
            frames.append(f)
            print(date, airports, len(f), int(f.category.eq("observed_at_assigned_stand").sum()), flush=True)
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
    elig["detector_taxi_s"] = mine.loc[elig[ID], "detector_taxi_s"].to_numpy(float)
    base = v56.set_index(ID).loc[elig[ID], "TAXITIME_SEC_mvt"].to_numpy(float)
    same = elig.comparator.to_numpy(float) == base
    checks["rows_where_comparator_differs_from_v56"] = int((~same).sum())
    elig = elig[same].reset_index(drop=True)

    train = S.extract(S.TRAIN, "train")
    train = train[train.category.eq("observed_at_assigned_stand")]
    train["cell"] = train.airport + "/" + train.date.str[5:7]
    gate = {c: drift(train[train.cell.eq(c)], elig[elig.cell.eq(c)]) for c in S.CELLS}
    kept = [c for c in S.CELLS if not gate[c]]
    elig = elig[elig.cell.isin(kept)].reset_index(drop=True)

    x = elig[S.COLS]
    corr = np.mean([R.joblib.load(S.CACHE / "models" / f"real_{s}.pkl").predict(x) for s in S.SEEDS], axis=0)
    pred = np.maximum(elig.comparator.to_numpy(float) + S.SHRINK * corr, 300.0)
    rev = x.iloc[::-1]
    replay = np.maximum(elig.comparator.to_numpy(float)[::-1] + S.SHRINK * np.mean(
        [R.joblib.load(S.CACHE / "models" / f"real_{s}.pkl").predict(rev) for s in S.SEEDS], axis=0), 300.0)[::-1]
    checks["reverse_order_replay_equal"] = bool(np.array_equal(pred, replay))

    cand = v56.copy()
    update = pd.Series(pred, index=elig[ID].to_numpy())
    hit = cand[ID].isin(update.index).to_numpy()
    cand.loc[hit, "TAXITIME_SEC_mvt"] = cand.loc[hit, ID].map(update).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand["TAXITIME_SEC_mvt"] = cand.TAXITIME_SEC_mvt.astype(template.TAXITIME_SEC_mvt.dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v56.loc[~hit].reset_index(drop=True))
    changed = int((cand.TAXITIME_SEC_mvt.to_numpy() != v56.TAXITIME_SEC_mvt.to_numpy()).sum())
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v56=bool(untouched),
                  all_finite=bool(np.isfinite(cand.TAXITIME_SEC_mvt).all()), rows_changed=changed,
                  no_row_on_days_6_13_20=bool(not elig.date.str[8:10].isin(NM_DAYS).any()))
    ok = not errors and untouched and checks["all_finite"] and checks["reverse_order_replay_equal"] and checks["no_row_on_days_6_13_20"]
    conf = json.loads((HERE / "research_320/results/stand_confirm.json").read_text())["arms"]["real"]["by_cell"]
    dev = json.loads((HERE / "research_320/results/stand_development.json").read_text())["arms"]["real"]["by_cell"]
    rows = elig.cell.value_counts().to_dict()
    plan = {}
    for name, src in (("confirmation", conf), ("development", dev)):
        mse = sum(rows.get(c, 0) * src[c]["gain_mse"] for c in kept) / S.N
        plan[name] = {"mse_per_scored_row": mse, "full_s": float(S.V56 - np.sqrt(S.V56 ** 2 - mse)),
                      "half_s": float(S.V56 - np.sqrt(S.V56 ** 2 - mse / 2))}
    report = {"checks": checks, "drift_gate": gate, "cells_kept": kept, "eligible_rows_by_cell": rows,
              "detector_estimates": int(elig.detector_taxi_s.notna().sum()),
              "shift_s": {q: float(np.percentile(pred - elig.comparator.to_numpy(float), p)) for q, p in (("p10", 10), ("median", 50), ("p90", 90))},
              "planning": plan, "base": {"version": "v56", "score": S.V56, "sha256": V56_SHA}}
    if ok:
        target = OUT / "candidate_v57_stand.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_sha256"] = sha(target)
        report["candidate_path"] = str(target)
    report["verdict"] = "built_and_verified" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("verdict", "checks", "cells_kept", "eligible_rows_by_cell", "planning")}, indent=1, default=str))


if __name__ == "__main__":
    main()
