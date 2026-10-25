"""Independent verification of candidate v57 (research_320/build_stand_candidate.py). Recomputes; imports no build code.

The seven checks the owner asked for, 27 September 2026:
  1 base      v57 is the exact scored v56 (SHA-256 1767cc02...) plus changed rows; its own hash matches the manifest
  2 preserved on every NM row, every EDDM (Munich) and EHAM (Amsterdam) row, every fallback row and every row v56
              changed against v55 (its partial-track overlay), v57 equals v56 exactly. NM rows are found without any NM
              file: v52 differs from v51 only through the NM add-on, so they are the rows where those two files differ
  3 no NM     no changed row is an NM row or lies on days 6, 13, 20; on every changed row v56 equals v52 there, i.e.
              the pre-NM value; the stand models were fitted on 2025 challenge rows and open tracks only
  4 IDs       v57, v56 and the submission template carry the same IDs in the same order
  5 replay    the 8,676 changed values are recomputed from the frozen models (hashes checked) and the cached features,
              in reverse order, and must equal v57 exactly
  6 validity  on the five kept cells, development and confirmation gains are positive in January and in July and
              positive after removing the best 100 rows
  7 new       v57 differs from v56 in more than zero rows and in its hash
Run from the repository root:  python3 -u -B research_320/verify_stand_candidate.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from research_320 import stand_experiment as S  # noqa: E402  (frozen models, cached features, detector definition)

ID = "MVT_ID_mvt"
V56 = HERE / ".cache/codex_v56/partial_release_v55/candidate.parquet"
V57_MANIFEST = HERE / "research_320/results/stand_candidate.json"
REL = HERE / ".cache/research_280/release"
OUT = HERE / "research_320/results/stand_candidate_verification.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(version):
    return json.loads((HERE / f"results/upbeat-goblin_{version}.json").read_text())


def gains(frame, cells):
    g = frame[(frame.airport + "/" + frame.date.str[5:7]).isin(cells) & frame.category.eq("observed_at_assigned_stand")]
    g = g.reset_index(drop=True)
    x = S.inputs(g, "real")
    corr = np.mean([S.R.joblib.load(S.CACHE / "models" / f"real_{s}.pkl").predict(x) for s in S.SEEDS], axis=0)
    pred = np.maximum(g.comparator.to_numpy(float) + S.SHRINK * corr, 300.0)
    y, base = g.y.to_numpy(float), g.comparator.to_numpy(float)
    gain = (base - y) ** 2 - (pred - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain, kind="stable")[-100:]] = False
    return {"rows": int(len(g)), "gain_mse": float(gain.mean()), "without_best100": float(gain[keep].mean()),
            "january": float(gain[g.date.str[5:7].eq("01").to_numpy()].mean()),
            "july": float(gain[g.date.str[5:7].eq("07").to_numpy()].mean())}


def main():
    m = json.loads(V57_MANIFEST.read_text())
    v57_path = Path(m["candidate_path"])
    v56, v57 = pd.read_parquet(V56), pd.read_parquet(v57_path)
    v55 = pd.read_parquet(HERE / ".cache/combined_candidate/candidate_v55_combined.parquet")
    v51 = pd.read_parquet(REL / "candidate_v51_combined.parquet")
    v52 = pd.read_parquet(REL / "candidate_v52_combined.parquet")
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    rank = pd.read_parquet(HERE / "data/ranking.parquet", columns=[ID, "ADEP_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt"])
    info = template[[ID]].merge(rank, on=ID, how="left", validate="one_to_one")
    day = pd.to_datetime(info.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%d").to_numpy()
    ap = info.ADEP_mvt.astype(str).to_numpy()
    fallback = info.AOBT_3_flt.isna().to_numpy()
    t = {k: f.TAXITIME_SEC_mvt.to_numpy() for k, f in (("v51", v51), ("v52", v52), ("v55", v55), ("v56", v56), ("v57", v57))}
    changed = t["v57"] != t["v56"]
    nm = t["v52"] != t["v51"]
    partial = t["v56"] != t["v55"]
    c = {}
    import subprocess
    official = json.loads(subprocess.run(["git", "show", "9884150:results/upbeat-goblin_v56.json"], cwd=HERE,
                                         capture_output=True, text=True, check=True).stdout)
    c["1_base"] = {"v56_official_score": official.get("score"),
                   "v56_sha": sha(V56) == "1767cc0257ddebf6350f1349d3c5487303ffa2df64f4ef0f9852db0b63e7c04c",
                   "v57_sha_matches_manifest": sha(v57_path) == m["candidate_sha256"],
                   "v51_v52_v55_files_match_records": all(sha(p) == record(v)["candidate_sha256"] for v, p in (
                       ("v51", REL / "candidate_v51_combined.parquet"), ("v52", REL / "candidate_v52_combined.parquet"),
                       ("v55", HERE / ".cache/combined_candidate/candidate_v55_combined.parquet")))}
    c["2_preserved"] = {"nm_rows_found": int(nm.sum()), "nm_rows_on_days_6_13_20_only": bool(np.isin(day[nm], ["06", "13", "20"]).all()),
                        "nm_rows_unchanged": bool((t["v57"][nm] == t["v56"][nm]).all()),
                        "munich_rows_unchanged": bool((t["v57"][ap == "EDDM"] == t["v56"][ap == "EDDM"]).all()),
                        "amsterdam_rows_unchanged": bool((t["v57"][ap == "EHAM"] == t["v56"][ap == "EHAM"]).all()),
                        "fallback_rows_unchanged": bool((t["v57"][fallback] == t["v56"][fallback]).all()),
                        "v56_partial_rows_found": int(partial.sum()),
                        "v56_partial_rows_unchanged": bool((t["v57"][partial] == t["v56"][partial]).all()),
                        "other_columns_identical": bool(v57.drop(columns="TAXITIME_SEC_mvt").equals(v56.drop(columns="TAXITIME_SEC_mvt")))}
    c["3_no_nm_inputs"] = {"changed_rows_not_nm": bool(not (changed & nm).any()),
                           "changed_rows_not_on_days_6_13_20": bool(not np.isin(day[changed], ["06", "13", "20"]).any()),
                           "v56_equals_pre_nm_v52_on_changed_rows": bool((t["v56"][changed] == t["v52"][changed]).all()),
                           "stand_models_trained_on": "2025 challenge training rows and adsb.lol tracks only (research_320/PREREG_stand.md)"}
    c["4_ids"] = {"v57_equals_template": bool(v57[ID].equals(template[ID])), "v57_equals_v56": bool(v57[ID].equals(v56[ID])),
                  "unique": bool(v57[ID].is_unique)}
    # 5: independent replay of every changed value, in reverse order
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet").set_index(ID)
    days = pd.concat([pd.read_parquet(p) for p in sorted((HERE / ".cache/research_320/stand_candidate").glob("day_*.parquet"))]).set_index(ID)
    ids = v57[ID].to_numpy()[changed][::-1]
    x = ctx.loc[ids, list(S.MS.COLS)].copy()
    x["detector_taxi_s"] = days.loc[ids, "detector_taxi_s"].to_numpy(float)
    man = json.loads((S.CACHE / "models/manifest.json").read_text())["model_hashes"]
    models = [S.R.joblib.load(S.CACHE / "models" / f"real_{s}.pkl") for s in S.SEEDS]
    hashes_ok = all(sha(S.CACHE / "models" / f"real_{s}.pkl") == man[f"real_{s}.pkl"] for s in S.SEEDS)
    pred = np.maximum(x.comparator.to_numpy(float) + S.SHRINK * np.mean([mo.predict(x[S.COLS]) for mo in models], axis=0), 300.0)
    got = pd.Series(t["v57"][changed][::-1])
    c["5_replay"] = {"model_hashes_ok": hashes_ok, "rows": int(changed.sum()),
                     "reverse_order_replay_exact": bool(np.array_equal(pred.astype(got.dtype), got.to_numpy())),
                     "max_abs_difference_s": float(np.max(np.abs(pred - got.to_numpy(float))))}
    kept = m["cells_kept"]
    dev, conf = gains(S.extract(S.STAGES["development"], "development"), kept), gains(S.extract(S.STAGES["confirm"], "confirm"), kept)
    c["6_validation_kept_cells"] = {"cells": kept, "development": dev, "confirmation": conf,
                                    "both_months_positive": all(v[k] > 0 for v in (dev, conf) for k in ("january", "july")),
                                    "positive_without_best100": all(v["without_best100"] > 0 for v in (dev, conf))}
    c["7_not_duplicate"] = {"rows_changed": int(changed.sum()), "hash_differs": m["candidate_sha256"] != sha(V56)}
    ok = (all(v for k, v in c["1_base"].items() if k != "v56_official_score") and all(v for k, v in c["2_preserved"].items() if isinstance(v, bool))
          and all(v for v in c["3_no_nm_inputs"].values() if isinstance(v, bool)) and all(c["4_ids"].values())
          and c["5_replay"]["model_hashes_ok"] and c["5_replay"]["reverse_order_replay_exact"]
          and c["6_validation_kept_cells"]["both_months_positive"] and c["6_validation_kept_cells"]["positive_without_best100"]
          and c["7_not_duplicate"]["rows_changed"] > 0 and c["7_not_duplicate"]["hash_differs"])
    report = {"candidate": "v57", "candidate_sha256": m["candidate_sha256"], "checks": c, "verdict": "verified" if ok else "FAILED"}
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
