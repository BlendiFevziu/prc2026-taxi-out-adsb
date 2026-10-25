"""Build v67 = the exact scored v66 with (a) the frozen stand model switched on for Zurich January and Amsterdam January
(PREREG_stand_january.md passed) and (b) the stand model at weight 1.0 instead of 0.5 on every row it corrects
(PREREG_stand_weight.md passed). NM-free. No upload.

Rows the stand model already corrects in v66 (owner "stand" in the v62 provenance, plus v66's LSZH and EHAM July rows)
must first reproduce their v66 value exactly as max(comparator + 0.5 x correction, 300); that proves the inputs equal
what shipped. New rows (LSZH and EHAM January, base-owned) must sit exactly at the comparator. Every one of these rows
becomes max(comparator + 1.0 x correction, 300). The 36th input comes from re-running Codex's extract_day with
summarise() extended, which must reproduce his 23 features and categories exactly (as in v57 and v66).
Run from the repository root:  python3 -u -B research_450_stand_january/build_v67.py
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
from research_320 import stand_experiment as S  # noqa: E402

MS, R = S.MS, S.R
import submission as SUB  # noqa: E402
from research_310 import full_coverage_2026 as F  # noqa: E402
from research_300 import geometry_compare as GC  # noqa: E402
import harness as H  # noqa: E402

ID, T, N, REF = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 256.4921
V57_CELLS = ("EDDF/01", "EDDF/07", "EGLL/01", "LEBL/01", "LEBL/07")
V66_CELLS = ("LSZH/07", "EHAM/07")
NEW_CELLS = ("LSZH/01", "EHAM/01")
C = HERE / ".cache/codex_v56"
V66 = HERE / ".cache/research_440/v66/candidate_v66.parquet"
CACHE = HERE / ".cache/research_450/stand_2026"
OUT = HERE / ".cache/research_450/v67"
REPORT = HERE / "research_450_stand_january/results/v67_build.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detector_inputs(wanted):
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
        with S.R.heavy_work("research_450: 2026 re-extraction for the stand model (all stand cells)", wait=True):
            for date in sorted(wanted.date.unique()):
                cache = CACHE / f"day_{date}.parquet"
                if not cache.exists():
                    airports = sorted(wanted.loc[wanted.date.eq(date), "airport"].unique())
                    day = dep[dep.date.eq(date) & dep.airport.isin(airports)]
                    F.extract_day(day, pd.read_parquet(R.MD.ADSB / f"tracks/{date}.parquet"), mapping, maps).to_parquet(cache, index=False)
                frames.append(pd.read_parquet(cache))
    finally:
        MS.summarise = original
    return pd.concat(frames, ignore_index=True).set_index(ID)


def main():
    rec66 = json.loads((HERE / "results/upbeat-goblin_v66.json").read_text())
    jan = json.loads((HERE / "research_450_stand_january/results/stand_january.json").read_text())
    wt = json.loads((HERE / "research_450_stand_january/results/stand_weight.json").read_text())
    checks = {"v66_is_scored_file": sha(V66) == rec66["file_sha256"], "january_passed": jan["verdict"] == "pass"
              and sorted(jan["passing_cells"]) == sorted(NEW_CELLS), "weight_passed": wt["verdict"] == "pass",
              "stand_models": S.models()["train_rows"] > 0}
    if not all(checks.values()):
        raise SystemExit(f"a precondition failed: {checks}")
    v66 = pd.read_parquet(V66)
    context = pd.read_parquet(C / "partial_release/context.parquet")
    codex = pd.read_parquet(C / "full_coverage_2026/features.parquet").set_index(ID)
    context["cell"] = context.airport + "/" + context.date.str[5:7]
    stand = context[context.category.eq("observed_at_assigned_stand") & context.cell.isin(V57_CELLS + V66_CELLS + NEW_CELLS)].copy()
    own = H.ownership_v62().set_index(ID).owner
    stand["owner"] = own.loc[stand[ID]].to_numpy()
    existing = stand[(stand.cell.isin(V57_CELLS) & stand.owner.eq("stand")) | (stand.cell.isin(V66_CELLS) & stand.owner.eq("base"))].copy()
    new = stand[stand.cell.isin(NEW_CELLS) & stand.owner.eq("base")].copy()
    base = H.current_base_2026()
    checks["comparator_equals_pre_nm_base"] = bool(np.array_equal(stand.comparator.to_numpy(float), base.loc[stand[ID]].to_numpy(float)))
    mine = detector_inputs(pd.concat([existing, new]))
    theirs = codex.loc[mine.index]
    cols = ["category"] + list(MS.FEATURES)
    checks["extraction_equals_codex"] = bool(all(
        np.array_equal(mine[c].to_numpy(), theirs[c].to_numpy()) if c == "category"
        else np.array_equal(mine[c].to_numpy(float), theirs[c].to_numpy(float), equal_nan=True) for c in cols))
    if not (checks["extraction_equals_codex"] and checks["comparator_equals_pre_nm_base"]):
        raise SystemExit(f"input gate failed: {checks}")
    models = [R.joblib.load(S.CACHE / "models" / f"real_{s}.pkl") for s in S.SEEDS]
    at66 = v66.set_index(ID)[T]

    def corr(frame):
        frame = frame.copy()
        frame["detector_taxi_s"] = mine.loc[frame[ID], "detector_taxi_s"].to_numpy(float)
        x = frame[S.COLS]
        c = np.mean([m.predict(x) for m in models], axis=0)
        c_rev = np.mean([m.predict(x.iloc[::-1]) for m in models], axis=0)[::-1]
        return c, bool(np.array_equal(c, c_rev))

    c_old, rep_old = corr(existing)
    c_new, rep_new = corr(new)
    checks["reverse_replay"] = rep_old and rep_new
    replay = np.maximum(existing.comparator.to_numpy(float) + 0.5 * c_old, 300.0).astype(v66[T].dtype)
    checks["existing_rows_by_cell"] = existing.cell.value_counts().to_dict()
    checks["existing_rows_reproduce_v66_at_0_5"] = bool(np.array_equal(replay, at66.loc[existing[ID]].to_numpy()))
    checks["new_rows_by_cell"] = new.cell.value_counts().to_dict()
    checks["new_rows_at_comparator"] = bool(np.array_equal(new.comparator.to_numpy(float).astype(v66[T].dtype), at66.loc[new[ID]].to_numpy()))
    if not (checks["reverse_replay"] and checks["existing_rows_reproduce_v66_at_0_5"] and checks["new_rows_at_comparator"]):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
        raise SystemExit(f"a replay gate failed: {checks}")
    update = pd.concat([pd.Series(np.maximum(existing.comparator.to_numpy(float) + c_old, 300.0), index=existing[ID].to_numpy()),
                        pd.Series(np.maximum(new.comparator.to_numpy(float) + c_new, 300.0), index=new[ID].to_numpy())])
    cand = v66.copy()
    hit = cand[ID].isin(update.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(update).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v66.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v66.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v66=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int((cand[T].to_numpy() != v66[T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    ok = not errors and untouched and checks["all_finite"]
    rows_new = {c: int(v) for c, v in checks["new_rows_by_cell"].items()}
    mse_new = sum(rows_new[c] * jan["cells"][c]["gain_mse"] for c in NEW_CELLS) / N          # measured at weight 0.5
    mse_weight = (len(existing) + len(new)) * wt["confirmation"]["gain_mse"] / N               # weight 1.0 over 0.5
    mse = mse_new + mse_weight
    at = lambda k: round(REF - (REF - np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v66", "score": rec66["score"], "sha256": rec66["file_sha256"]},
              "prediction_fixed_before_upload": {"assumptions": "January cells at their weight-0.5 test gain plus the weight test's confirmation gain on all "
                                                                "stand rows; central 0.8, worst 0.3, best 1.2 of full transfer",
                                                 "mse_per_scored_row_full": mse, "central": at(0.8), "worst": at(0.3), "best": at(1.2)}}
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v67.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
