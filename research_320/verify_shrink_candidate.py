"""Independent seven-point verification of candidate v58 (research_320/build_shrink_candidate.py); imports no build code.

  1 base      v58 is the scored v57 (SHA-256 88245bfc..., 260.9292) plus changed rows; its own hash matches the manifest
  2 preserved on every NM row (v52 against v51, no NM file read), every EDDM row, every guarded Amsterdam detector row
              (EHAM rows where v55 differs from v52), every fallback row and every stand row of v57, v58 equals v57
  3 no NM     the changed rows are all inside Codex's partial-track overlay and none is an NM row; the corrections come
              from the partial-track models fitted on 2025 challenge rows and open tracks
  4 IDs       same IDs, same order as v57 and the submission template
  5 replay    every changed value recomputed from the rebuilt models (hash-checked) on the 2026 context, reverse order
  6 validity  s = 1.0 against 0.5: development-test and fresh-date gains positive in January and July and positive after
              removing the best 100 rows
  7 new       v58 differs from v57 in more than zero rows and in its hash
Run from the repository root:  python3 -u -B research_320/verify_shrink_candidate.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import stand_experiment as S  # noqa: E402  (frozen modules and seeds only)
from research_320 import audit_v56_errors as A  # noqa: E402  (v56's development cohort definition)

ID = "MVT_ID_mvt"
REL = HERE / ".cache/research_280/release"
REBUILD = HERE / ".cache/research_320/v56_rebuild"
C = HERE / ".cache/codex_v56"
OUT = HERE / "research_320/results/shrink_candidate_verification.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(g, month):
    keep = np.ones(len(g), bool)
    keep[np.argsort(g, kind="stable")[-100:]] = False
    return {"rows": int(len(g)), "gain_mse": float(g.mean()), "without_best100": float(g[keep].mean()),
            "january": float(g[month == "01"].mean()), "july": float(g[month == "07"].mean())}


def main():
    m = json.loads((HERE / "research_320/results/shrink_candidate.json").read_text())
    rec57 = json.loads((HERE / "results/upbeat-goblin_v57.json").read_text())
    v57_path = HERE / ".cache/research_320/stand_candidate/candidate_v57_stand.parquet"
    files = {"v51": REL / "candidate_v51_combined.parquet", "v52": REL / "candidate_v52_combined.parquet",
             "v55": HERE / ".cache/combined_candidate/candidate_v55_combined.parquet",
             "v56": C / "partial_release_v55/candidate.parquet", "v57": v57_path, "v58": Path(m["candidate_path"])}
    f = {k: pd.read_parquet(p) for k, p in files.items()}
    t = {k: v.TAXITIME_SEC_mvt.to_numpy() for k, v in f.items()}
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    rank = pd.read_parquet(HERE / "data/ranking.parquet", columns=[ID, "ADEP_mvt", "AOBT_3_flt"])
    info = template[[ID]].merge(rank, on=ID, how="left", validate="one_to_one")
    ap, fallback = info.ADEP_mvt.astype(str).to_numpy(), info.AOBT_3_flt.isna().to_numpy()
    changed, nm = t["v58"] != t["v57"], t["v52"] != t["v51"]
    detector = (ap == "EHAM") & (t["v55"] != t["v52"])
    stand = t["v57"] != t["v56"]
    upd = pd.read_parquet(C / "partial_release_v55/updates.parquet")
    overlay = f["v58"][ID].isin(upd[ID]).to_numpy()
    c = {"1_base": {"v57_sha_is_scored": sha(v57_path) == rec57["file_sha256"], "v57_official": rec57["score"],
                    "v58_sha_matches_manifest": sha(files["v58"]) == m["candidate_sha256"]},
         "2_preserved": {"nm_rows": int(nm.sum()), "nm_unchanged": bool((t["v58"][nm] == t["v57"][nm]).all()),
                         "munich_unchanged": bool((t["v58"][ap == "EDDM"] == t["v57"][ap == "EDDM"]).all()),
                         "amsterdam_detector_rows": int(detector.sum()),
                         "amsterdam_detector_unchanged": bool((t["v58"][detector] == t["v57"][detector]).all()),
                         "fallback_unchanged": bool((t["v58"][fallback] == t["v57"][fallback]).all()),
                         "v57_stand_rows": int(stand.sum()), "v57_stand_unchanged": bool((t["v58"][stand] == t["v57"][stand]).all()),
                         "other_columns_identical": bool(f["v58"].drop(columns="TAXITIME_SEC_mvt").equals(f["v57"].drop(columns="TAXITIME_SEC_mvt")))},
         "3_no_nm_inputs": {"changed_inside_partial_overlay": bool(not (changed & ~overlay).any()),
                            "changed_rows_not_nm": bool(not (changed & nm).any()),
                            "models": "v56 partial-track real models, rebuilt byte-identical to Codex's; 2025 challenge rows and adsb.lol tracks"},
         "4_ids": {"v58_equals_template": bool(f["v58"][ID].equals(template[ID])), "v58_equals_v57": bool(f["v58"][ID].equals(f["v57"][ID])),
                   "unique": bool(f["v58"][ID].is_unique)}}
    ctx = pd.read_parquet(C / "partial_release/context.parquet").set_index(ID)
    ids = f["v58"][ID].to_numpy()[changed][::-1]
    x = ctx.loc[ids, list(S.MS.COLS)]
    rebuilt = json.loads((REBUILD / "multiairport_screen_rebuilt.json").read_text())["model_hashes"]
    hashes_ok = all(sha(REBUILD / f"real_{s}.pkl") == rebuilt[f"real_{s}.pkl"] for s in S.SEEDS)
    corr = np.mean([S.R.joblib.load(REBUILD / f"real_{s}.pkl").predict(x) for s in S.SEEDS], axis=0)
    pred = np.maximum(x.comparator.to_numpy(float) + 1.0 * corr, 300.0)
    got = t["v58"][changed][::-1]
    c["5_replay"] = {"model_hashes_ok": hashes_ok, "rows": int(changed.sum()),
                     "reverse_order_replay_exact": bool(np.array_equal(pred.astype(got.dtype), got)),
                     "max_abs_difference_s": float(np.max(np.abs(pred - got.astype(float))))}
    pr = pd.read_parquet(REBUILD / "predictions.parquet")
    pr["month"] = pr.date.str[5:7]
    cell = np.array([(a, mm) in A.DEPLOYED for a, mm in zip(pr.airport, pr.month)])
    p = pr[cell & pr.category.eq("partial_ground").to_numpy() & ~pr.MVT_ID_mvt.isin(A.detector_rows()).to_numpy()].reset_index(drop=True)
    dc = np.mean([S.R.joblib.load(REBUILD / f"real_{s}.pkl").predict(p[list(S.MS.COLS)]) for s in S.SEEDS], axis=0)
    y, base = p.y.to_numpy(float), p.comparator.to_numpy(float)
    dev = summary((np.maximum(base + .5 * dc, 300.) - y) ** 2 - (np.maximum(base + dc, 300.) - y) ** 2, p.month.to_numpy())
    fresh = json.loads((HERE / "research_320/results/shrink_confirm.json").read_text())["A_v56_partial"]["real"]
    c["6_validation"] = {"development_test_dates": dev,
                         "fresh_dates": {"rows": fresh["rows"], "gain_mse": fresh["gain_mse"], "without_best100": fresh["without_best100"],
                                         "january": fresh["by_month"]["01"], "july": fresh["by_month"]["07"]}}
    c["6_validation"]["both_months_positive"] = all(v[k] > 0 for v in c["6_validation"].values() if isinstance(v, dict) for k in ("january", "july"))
    c["6_validation"]["positive_without_best100"] = all(v["without_best100"] > 0 for v in c["6_validation"].values() if isinstance(v, dict) and "without_best100" in v)
    c["7_not_duplicate"] = {"rows_changed": int(changed.sum()), "hash_differs": m["candidate_sha256"] != rec57["file_sha256"]}
    ok = (c["1_base"]["v57_sha_is_scored"] and c["1_base"]["v58_sha_matches_manifest"]
          and all(v for v in c["2_preserved"].values() if isinstance(v, bool))
          and all(v for v in c["3_no_nm_inputs"].values() if isinstance(v, bool)) and all(c["4_ids"].values())
          and c["5_replay"]["model_hashes_ok"] and c["5_replay"]["reverse_order_replay_exact"]
          and c["6_validation"]["both_months_positive"] and c["6_validation"]["positive_without_best100"]
          and c["7_not_duplicate"]["rows_changed"] > 0 and c["7_not_duplicate"]["hash_differs"])
    report = {"candidate": "v58", "candidate_sha256": m["candidate_sha256"], "checks": c, "verdict": "verified" if ok else "FAILED"}
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
