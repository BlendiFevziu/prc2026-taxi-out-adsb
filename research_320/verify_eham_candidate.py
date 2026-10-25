"""Independent seven-point verification of candidate v59 (research_320/build_eham_candidate.py); imports no build code.

  1 base      v59 is the scored v58 (826e4609..., 259.9035) plus changed rows; its hash matches the manifest
  2 preserved v59 equals v58 on every NM row (v52 against v51), EDDM row, guarded Amsterdam detector row, fallback row,
              v57 stand row, v58 partial-track row, and in every column but the prediction
  3 no NM     the changed rows lie inside Codex's EHAM January overlay, none is an NM row; his add-on models were
              fitted on 2025 challenge rows and open tracks
  4 IDs       same IDs, same order as v58 and the submission template
  5 replay    every changed value recomputed from his models (hash-checked) on his 2026 inputs, reverse order
  6 validity  fresh January days: the add-on at s = 1.0 beats no add-on and s = 0.5, also without the best 100; on his
              own screen and confirmation dates s = 1.0 beats 0.5 as well (the add-on is January-only by design)
  7 new       rows changed and hash differ from v58
Run from the repository root:  python3 -u -B research_320/verify_eham_candidate.py
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
import main_model  # noqa: F401,E402
import joblib  # noqa: E402

ID = "MVT_ID_mvt"
C = HERE / ".cache/codex_eham"
REL = HERE / ".cache/research_280/release"
SCHEMA = ["comparator", "ersatz_s", "schedule", "gate", "hour_sin", "hour_cos", "first_age_s", "last_age_s", "span_s",
          "endpoint_distance_m", "weighted_speed_kt", "weighted_still_share", "observed_interval_share", "is_july"]
OUT = HERE / "research_320/results/eham_candidate_verification.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    m = json.loads((HERE / "research_320/results/eham_candidate.json").read_text())
    rec58 = json.loads((HERE / "results/upbeat-goblin_v58.json").read_text())
    paths = {"v51": REL / "candidate_v51_combined.parquet", "v52": REL / "candidate_v52_combined.parquet",
             "v55": HERE / ".cache/combined_candidate/candidate_v55_combined.parquet",
             "v56": HERE / ".cache/codex_v56/partial_release_v55/candidate.parquet",
             "v57": HERE / ".cache/research_320/stand_candidate/candidate_v57_stand.parquet",
             "v58": HERE / ".cache/research_320/shrink_candidate/candidate_v58_shrink.parquet", "v59": Path(m["candidate_path"])}
    f = {k: pd.read_parquet(p) for k, p in paths.items()}
    t = {k: v.TAXITIME_SEC_mvt.to_numpy() for k, v in f.items()}
    template = pd.read_parquet(HERE / "data/submitting.parquet")
    info = template[[ID]].merge(pd.read_parquet(HERE / "data/ranking.parquet", columns=[ID, "ADEP_mvt", "AOBT_3_flt"]), on=ID, how="left")
    ap, fallback = info.ADEP_mvt.astype(str).to_numpy(), info.AOBT_3_flt.isna().to_numpy()
    changed, nm = t["v59"] != t["v58"], t["v52"] != t["v51"]
    detector = (ap == "EHAM") & (t["v55"] != t["v52"])
    stand, partial58 = t["v57"] != t["v56"], t["v58"] != t["v57"]
    upd = pd.read_parquet(C / "eham_on_scored_v56/updates.parquet")
    overlay = f["v59"][ID].isin(upd[ID]).to_numpy()
    same = lambda k: bool((t["v59"][k] == t["v58"][k]).all())
    c = {"1_base": {"v58_is_scored": sha(paths["v58"]) == rec58["file_sha256"], "v58_official": rec58["score"],
                    "v59_matches_manifest": sha(paths["v59"]) == m["candidate_sha256"]},
         "2_preserved": {"nm_rows": int(nm.sum()), "nm_unchanged": same(nm), "munich_unchanged": same(ap == "EDDM"),
                         "amsterdam_detector_rows": int(detector.sum()), "amsterdam_detector_unchanged": same(detector),
                         "fallback_unchanged": same(fallback), "v57_stand_unchanged": same(stand),
                         "v58_partial_rows": int(partial58.sum()), "v58_partial_unchanged": same(partial58),
                         "other_columns_identical": bool(f["v59"].drop(columns="TAXITIME_SEC_mvt").equals(f["v58"].drop(columns="TAXITIME_SEC_mvt")))},
         "3_no_nm_inputs": {"changed_inside_eham_overlay": bool(not (changed & ~overlay).any()), "changed_rows_not_nm": bool(not (changed & nm).any()),
                            "changed_rows_all_EHAM_January": bool(set(ap[changed]) == {"EHAM"}),
                            "models": "Codex's EHAM add-on (8d98953), 2025 challenge rows and adsb.lol tracks"},
         "4_ids": {"v59_equals_template": bool(f["v59"][ID].equals(template[ID])), "v59_equals_v58": bool(f["v59"][ID].equals(f["v58"][ID]))}}
    hashes = json.loads(subprocess.run(["git", "show", "8d98953:research_310/results/eham_recovery_screen.json"], cwd=HERE,
                                       capture_output=True, text=True, check=True).stdout)["model_hashes"]
    feats = pd.concat([pd.read_parquet(p) for p in sorted((C / "eham_recovery").glob("features_2026-01-*.parquet"))], ignore_index=True).set_index(ID)
    ctx = pd.read_parquet(HERE / ".cache/codex_v56/partial_release/context.parquet").set_index(ID)
    ids = f["v59"][ID].to_numpy()[changed][::-1]
    x = pd.concat([ctx.loc[ids, SCHEMA[:6] + ["is_july"]], feats.loc[ids, SCHEMA[6:13]]], axis=1)[SCHEMA]
    models = [joblib.load(C / "eham_recovery" / f"real_{s}.pkl") for s in (177, 277, 377)]
    pred = np.maximum(x.comparator.to_numpy(float) + np.mean([mo.predict(x) for mo in models], axis=0), 300.0)
    got = t["v59"][changed][::-1]
    c["5_replay"] = {"model_hashes_ok": all(sha(C / "eham_recovery" / f"real_{s}.pkl") == hashes[f"real_{s}.pkl"] for s in (177, 277, 377)),
                     "rows": int(changed.sum()), "reverse_order_replay_exact": bool(np.array_equal(pred.astype(got.dtype), got)),
                     "max_abs_difference_s": float(np.max(np.abs(pred - got.astype(float))))}
    fresh = json.loads((HERE / "research_320/results/eham_weight.json").read_text())
    dev = json.loads((HERE / "research_320/results/eham_weight_headroom.json").read_text())
    c["6_validation"] = {"fresh_value_vs_no_addon": {k: fresh["comparison_2_value"][k] for k in ("rows", "gain_mse", "without_best100")},
                         "fresh_weight_1_vs_0_5": {k: fresh["comparison_1_weight"][k] for k in ("rows", "gain_mse", "without_best100")},
                         "codex_dates_weight_1_vs_0_5": {st: dev[st]["curve"]["1.0"] - dev[st]["curve"]["0.5"] for st in ("screen", "confirmation")},
                         "scope": "January only, by the add-on's design; EHAM July is v56's own cell"}
    v = c["6_validation"]
    v["positive"] = (v["fresh_value_vs_no_addon"]["gain_mse"] > 0 and v["fresh_value_vs_no_addon"]["without_best100"] > 0
                     and v["fresh_weight_1_vs_0_5"]["gain_mse"] > 0 and v["fresh_weight_1_vs_0_5"]["without_best100"] > 0
                     and all(g > 0 for g in v["codex_dates_weight_1_vs_0_5"].values()))
    c["7_not_duplicate"] = {"rows_changed": int(changed.sum()), "hash_differs": m["candidate_sha256"] != rec58["file_sha256"]}
    ok = (c["1_base"]["v58_is_scored"] and c["1_base"]["v59_matches_manifest"]
          and all(x for x in c["2_preserved"].values() if isinstance(x, bool))
          and all(x for x in c["3_no_nm_inputs"].values() if isinstance(x, bool)) and all(c["4_ids"].values())
          and c["5_replay"]["model_hashes_ok"] and c["5_replay"]["reverse_order_replay_exact"] and v["positive"]
          and c["7_not_duplicate"]["rows_changed"] > 0 and c["7_not_duplicate"]["hash_differs"])
    report = {"candidate": "v59", "candidate_sha256": m["candidate_sha256"], "checks": c, "verdict": "verified" if ok else "FAILED"}
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
