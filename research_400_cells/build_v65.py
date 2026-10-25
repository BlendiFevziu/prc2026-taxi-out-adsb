"""Build v65 = the exact scored v64 with the frozen C1 model switched on for Munich July, if it passes
research_400_cells/PREREG_cells2.md (rules 1 to 4 evaluated here before anything is written). NM-free. No upload.

Reuses build_v64's 2026 context recipe (proven bit for bit against Codex's context rows). Only base-owned partial rows
that sit exactly at the comparator in v64; the new value is max(comparator + C1, 300).
Run from the repository root:  python3 -u -B research_400_cells/build_v65.py
"""
import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "research_400_cells"))
import build_v64 as B  # noqa: E402

MS, H, L, SUB = B.MS, B.H, B.L, B.SUB
ID, T, N, REF, CELL = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841, 257.0986, "EDDM/07"
V64 = HERE / ".cache/research_400/v64/candidate_v64.parquet"
OUT = HERE / ".cache/research_400/v65"
REPORT = HERE / "research_400_cells/results/v65_build.json"


def rules(rows):
    """PREREG_cells2 rules 1 to 4 for EDDM/07 against its sister month EDDM/01."""
    from error_by_track_quality import stratum
    conf = json.loads((HERE / "research_400_cells/results/cells_confirm.json").read_text())["cells"]["EDDM/01"]
    src = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(HERE / ".cache/research_400/days/EDDM_2025-01-*.parquet")))], ignore_index=True)
    src = src[src.category.eq("partial_ground")]
    flags = []
    for c in MS.FEATURES:
        a, b = src[c].astype(float), rows[c].astype(float)
        sd = a.std()
        if (abs(b.mean() - a.mean()) / sd if sd > 0 else 0.0) > 1.0 or abs(b.isna().mean() - a.isna().mean()) > 0.2:
            flags.append(c)
    mix = pd.Series(stratum(rows).to_numpy()).value_counts(normalize=True)
    gains = conf["by_stratum"]
    big = {k: v for k, v in mix.items() if v >= 0.05}
    rule3 = all(k in gains and gains[k]["rows"] >= 30 and gains[k]["gain_mse"] > 0 for k in big)
    weighted = sum(v * gains[k]["gain_mse"] for k, v in mix.items() if k in gains and gains[k]["rows"] >= 30)
    mse = len(rows) * weighted / N
    full_s = float(REF - np.sqrt(REF ** 2 - mse)) if mse > 0 else float(mse / (2 * REF))
    return {"1_sister_month_passed": conf["verdict"] == "pass", "2_no_drift_flag": not flags, "3_large_strata_positive": rule3,
            "4_planning_at_least_0_1_s": full_s >= 0.1}, {"drift_flags": flags, "mix": mix.round(4).to_dict(), "weighted_gain": weighted,
                                                          "mse_per_scored_row_full": mse, "full_s": full_s}


def main():
    rec64 = json.loads((HERE / "results/upbeat-goblin_v64.json").read_text())
    checks = {"v64_is_scored_file": B.sha(V64) == rec64["file_sha256"]}
    if not all(checks.values()):
        raise SystemExit(f"precondition failed: {checks}")
    feats = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(B.AUDIT / "day_2026-*.parquet")))], ignore_index=True)
    feats["cell"] = feats.airport + "/" + feats.date.str[5:7]
    rows = feats[feats.cell.eq(CELL) & feats.category.eq("partial_ground")].drop(columns="cell").reset_index(drop=True)
    own = H.ownership_v62().set_index(ID).owner          # v63/v64 changed other cells only
    owner = own.loc[rows[ID]].to_numpy()
    checks["partial_rows_by_owner"] = pd.Series(owner).value_counts().to_dict()
    rows = rows[owner == "base"].reset_index(drop=True)
    verdict_rules, detail = rules(rows)
    checks["rules"], checks["rule_detail"] = verdict_rules, detail
    if not all(verdict_rules.values()):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "cell_fails_registration"}, indent=2, default=str) + "\n")
        raise SystemExit(f"EDDM/07 fails the registration: {verdict_rules}")
    frame = B.context_2026(rows)
    v64 = pd.read_parquet(V64)
    at64 = v64.set_index(ID).loc[frame[ID], T].to_numpy()
    checks["v64_rows_at_comparator"] = bool(np.array_equal(frame.comparator.to_numpy(float).astype(at64.dtype), at64))
    corr = L.correction("partial", frame)
    rev = L.correction("partial", frame.iloc[::-1].reset_index(drop=True))[::-1]
    checks["c1_reverse_replay"] = bool(np.array_equal(corr, rev))
    if not (checks["v64_rows_at_comparator"] and checks["c1_reverse_replay"]):
        REPORT.write_text(json.dumps({"checks": checks, "verdict": "gate_failed"}, indent=2, default=str) + "\n")
        raise SystemExit(f"a replay gate failed: {checks}")
    new = pd.Series(L.apply(frame, corr, 1.0), index=frame[ID].to_numpy())
    cand = v64.copy()
    hit = cand[ID].isin(new.index).to_numpy()
    cand.loc[hit, T] = cand.loc[hit, ID].map(new).to_numpy(float)
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    cand[T] = cand[T].astype(template[T].dtype)
    errors = SUB.pruefe(cand, template, None)
    untouched = cand.loc[~hit].reset_index(drop=True).equals(v64.loc[~hit].reset_index(drop=True))
    change = cand.loc[hit, T].to_numpy(float) - v64.loc[hit, T].to_numpy(float)
    checks.update(pruefe_errors=[str(e) for e in errors], untouched_rows_identical_to_v64=bool(untouched),
                  all_finite=bool(np.isfinite(cand[T]).all()), rows_changed=int((cand[T].to_numpy() != v64[T].to_numpy()).sum()),
                  change_s={"mean": float(change.mean()), "rms": float(np.sqrt(np.mean(change ** 2)))})
    ok = not errors and untouched and checks["all_finite"]
    mse = detail["mse_per_scored_row_full"]
    at = lambda k: round(REF - (REF - np.sqrt(REF ** 2 - k * mse)), 2)
    report = {"checks": checks, "base": {"version": "v64", "score": rec64["score"], "sha256": rec64["file_sha256"]},
              "prediction_fixed_before_upload": {"assumptions": "sister-month per-stratum gains weighted to the 2026 mix (no same-month confirmation); "
                                                                "central 0.8 of full transfer, worst 0.3, best 1.2 (v63 realised 1.46 and v64 1.16 of their "
                                                                "stratum or registered values)", "mse_per_scored_row_full": mse,
                                                 "central": at(0.8), "worst": at(0.3), "best": at(1.2)}}
    if ok:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "candidate_v65.parquet"
        cand.to_parquet(target, index=False)
        report["candidate_path"], report["candidate_sha256"] = str(target), B.sha(target)
    report["verdict"] = "built" if ok else "gate_failed"
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
