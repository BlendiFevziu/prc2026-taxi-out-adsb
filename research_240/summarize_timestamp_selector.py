"""Recompute a timestamp-selector report from saved fits/predictions, without fitting.

Also recovers the 13 September run that completed all six pairs but failed exporting
a NumPy integer to JSON. Verifies input hashes, saved coefficient replay, baseline
parity and all twelve optimizer completion records. No new selection or tuning.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from research_240.timestamp_selector import (CANDIDATES, PAIRS, RIDGES, SEEDS, STRENGTHS,
    Design, candidate_matrix, metrics, weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-project", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=Path(".cache/research_240/timestamp_selector_source_split"))
    parser.add_argument("--log", type=Path, default=Path(".cache/research_240/timestamp_selector_source_split.log"))
    parser.add_argument("--report", type=Path, default=Path("research_240/results/timestamp_selector.json"))
    args = parser.parse_args()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((args.run/"manifest.json").read_text())
    for file, digest in manifest["input_sha256"].items():
        if sha(args.cache_project/file) != digest:
            raise ValueError(f"changed input {file}")
    for file, digest in manifest["source_data_sha256"].items():
        if sha(Path(file)) != digest:
            raise ValueError(f"changed source {file}")
    for file, digest in manifest["production_code_sha256"].items():
        if sha(Path(file)) != digest:
            raise ValueError(f"changed baseline code {file}")
    cache = args.cache_project/".cache/research_240"
    index = {tuple(sorted(json.loads(p.read_text())["source_months"])):p.with_suffix(".parquet")
             for p in (cache/"rescreen_production").glob("rescreen_*.json")}
    fits = [{"pair":p,"ridge":float(r),"iterations":int(n),"success":s=="True"}
            for p,r,n,s in re.findall(r"GATE (\d+_\d+) ridge=([\d.]+) iterations=(\d+) success=(True|False)",args.log.read_text())]
    if len(fits) != 12 or len({(f["pair"],f["ridge"]) for f in fits}) != 12:
        raise ValueError("expected twelve distinct optimizer completion records")
    frames, masks, diagnostics = [], [], []
    for pair in PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        va = pd.read_parquet(index[pair]).reset_index(drop=True)
        frame = pd.read_parquet(args.run/f"predictions_{key}.parquet")
        parts = pd.read_parquet(args.run/f"outer_parts_{key}.parquet")
        ref = pd.read_parquet(cache/"fallback_anchor_production"/f"predictions_{key}.parquet").set_index("MVT_ID_mvt").loc[va.MVT_ID_mvt]
        for other in (frame,parts):
            np.testing.assert_array_equal(va.MVT_ID_mvt.to_numpy(),other.MVT_ID_mvt.to_numpy())
        np.testing.assert_array_equal(va.y.to_numpy(),frame.y.to_numpy())
        np.testing.assert_array_equal(ref.y.to_numpy(),frame.y.to_numpy())
        np.testing.assert_allclose(frame.baseline, ref.prod_anchor, rtol=0,atol=1e-6)
        cv,mv = candidate_matrix(va,parts)
        replay_max = 0.0
        for ridge in RIDGES:
            saved = json.loads((args.run/f"gate_{key}_ridge{ridge}.json").read_text())
            design = Design()
            design.mean,design.std = np.array(saved["mean"]),np.array(saved["std"])
            design.airports = saved["airports"]
            pred = (weights(np.array(saved["theta"]),design.transform(va,parts),mv)*cv).sum(axis=1)
            for strength in STRENGTHS:
                fresh = frame.baseline+strength*(pred-frame.baseline)
                diff = float(np.abs(fresh-frame[f"ridge{ridge}_blend{strength}"]).max())
                if diff > 1e-6:
                    raise ValueError(f"saved selector replay mismatch {key}: {diff}")
                replay_max = max(replay_max,diff)
        y = va.y.to_numpy(float)
        errors = np.where(mv,(cv-y[:,None])**2,np.inf)
        low,high = np.where(mv,cv,np.inf).min(axis=1),np.where(mv,cv,-np.inf).max(axis=1)
        diagnostics.append({"pair":key,"rows":len(va),"selector_replay_max_s":replay_max,
            "baseline_parity_max_s":float(np.abs(frame.baseline.to_numpy()-ref.prod_anchor.to_numpy()).max()),
            "baseline_sse":float(((frame.baseline-y)**2).sum()),
            "best_single_oracle_sse":float(errors.min(axis=1).sum()),
            "convex_oracle_sse":float(((np.clip(y,low,high)-y)**2).sum())})
        frames.append(frame)
        masks.append(mv)
    d = pd.concat(frames,ignore_index=True)
    if not d.MVT_ID_mvt.is_unique:
        raise ValueError("duplicate pooled held-out IDs")
    ranking = pd.read_parquet(next((cache/"anchor_preview").glob("fallback_2026_ranking_*.parquet")))
    dummy = pd.DataFrame({"baseline":np.full(len(ranking),1200),"normal":np.full(len(ranking),1200),"high":np.full(len(ranking),6000)})
    _,rm = candidate_matrix(ranking,dummy)
    cvg = np.concatenate(masks).mean(axis=0)
    coverage = {name:{"validation":float(cvg[i]),"ranking":float(rm[:,i].mean())}
                for i,name in enumerate(CANDIDATES) if i>=3}
    coverage_pass = all(abs(v["validation"]-v["ranking"]) <= .05 for v in coverage.values())
    full = pd.read_parquet(args.cache_project/".cache/clock_queue_release_20260912/candidate/holdout_predictions.parquet").set_index("MVT_ID_mvt")
    jan_jul = d[d.fold.eq("1_7")].set_index("MVT_ID_mvt")
    np.testing.assert_array_equal(full.loc[jan_jul.index,"y"],jan_jul.y)
    base = full.prediction.copy()
    base.loc[jan_jul.index] = jan_jul.baseline
    baseline_overall = float(np.sqrt(np.mean((base-full.y)**2)))
    report = {"manifest":manifest,"recovery":"all fitting completed; report-only recovery after numpy.int64 JSON export error; no refits",
        "summary_code_sha256":sha(Path(__file__)), "rows":len(d),"baseline_rmse":float(np.sqrt(np.mean((d.baseline-d.y)**2))),
        "diagnostics":diagnostics,"fits":fits,"candidate_coverage":coverage,"coverage_pass":coverage_pass,
        "january_july_overall_baseline_rmse":baseline_overall,"variants":{}}
    for ridge in RIDGES:
        for strength in STRENGTHS:
            name = f"ridge{ridge}_blend{strength}"
            m = metrics(d,d[name].to_numpy())
            controls = {str(s):metrics(d,d[f"{name}_perm{s}"].to_numpy())["mse_gain"] for s in SEEDS}
            m["permutation_mse_gains"] = controls
            m["optimizer_converged"] = all(f["success"] for f in fits if f["ridge"]==ridge)
            m["passes_screen"] = bool(m["mse_gain"]>0 and m["positive_pairs"]>=4 and m["without_top100_mse_gain"]>0
                and m["january_mse_gain"]>0 and m["july_mse_gain"]>0 and m["mse_gain"]>max(controls.values())
                and coverage_pass and m["optimizer_converged"])
            pred = base.copy()
            pred.loc[jan_jul.index] = jan_jul[name]
            m["january_july_overall_rmse"] = float(np.sqrt(np.mean((pred-full.y)**2)))
            m["january_july_overall_rmse_gain"] = baseline_overall-m["january_july_overall_rmse"]
            report["variants"][name] = m
    report["decision"] = "ADVANCE_TO_INDEPENDENT_CHECKS" if any(v["passes_screen"] for v in report["variants"].values()) else "REJECT_KEEP_V46"
    report["artifact_sha256"] = {p.name:sha(p) for p in args.run.iterdir() if p.is_file() and p.name != "pooled_predictions.parquet"}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False))
    d.to_parquet(args.run/"pooled_predictions.parquet",index=False)
    print(json.dumps({k:report[k] for k in ("rows","baseline_rmse","january_july_overall_baseline_rmse","variants","decision")},indent=2))


if __name__ == "__main__":
    main()
