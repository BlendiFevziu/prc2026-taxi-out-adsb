"""Replay saved main selectors and compare training/evaluation errors; no fitting.

This diagnostic is run after the registered screen. It verifies saved model files,
documents transfer failure and checks raw clock arithmetic independently. It does
not select new settings, remove rows, or generate competition predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_240.main_timestamp_selector import (CATEGORICAL,CLOCKS,NUMERIC,PRIORS,
    Design,candidates,predict_linear,predict_neural,source_months,write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-project",type=Path,required=True)
    parser.add_argument("--run",type=Path,default=Path(".cache/research_240/main_timestamp_selector"))
    parser.add_argument("--screen-report",type=Path,default=Path("research_240/results/main_timestamp_selector.json"))
    parser.add_argument("--output",type=Path,default=Path("research_240/results/main_timestamp_selector_audit.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    project = args.cache_project.resolve()
    release = project/".cache/clock_queue_release_20260912"
    screen = json.loads(args.screen_report.read_text())
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,digest in screen["artifact_sha256"].items():
        if sha(args.run/name)!=digest:
            raise ValueError(f"changed saved artifact {name}")
    for name,digest in screen["manifest"]["input_sha256"].items():
        if sha(project/name)!=digest:
            raise ValueError(f"changed input {name}")
    if sha(Path("research_240/main_timestamp_selector.py"))!=screen["manifest"]["code_sha256"]:
        raise ValueError("fitting source changed")
    cols = list(dict.fromkeys(list(NUMERIC)+list(CLOCKS)+list(CATEGORICAL)+["hat_aobt","MVT_ID_mvt","MVT_TIME_UTC_mvt","y"]))
    frame = pd.read_parquet(release/"data/months_1_7.parquet",columns=cols)
    frame = frame[frame.hat_aobt.eq(1)].reset_index(drop=True)
    base = pd.read_parquet(release/"candidate/holdout_predictions.parquet").set_index("MVT_ID_mvt",verify_integrity=True).loc[frame.MVT_ID_mvt,"prediction"].to_numpy()
    views = pd.read_parquet(project/".cache/research_240/main_views/predictions_outer.parquet").set_index("MVT_ID_mvt",verify_integrity=True).loc[frame.MVT_ID_mvt]
    pooled = pd.read_parquet(args.run/"pooled_predictions.parquet").set_index("MVT_ID_mvt",verify_integrity=True).loc[frame.MVT_ID_mvt]
    paths = {m:next((project/"data").glob(f"training_2025-{m:02d}-01_*.parquet")) for m in (1,7)}
    source = source_months(frame,paths)
    np.testing.assert_array_equal(source,pooled.source_month)
    np.testing.assert_array_equal(frame.y,pooled.y)
    c,mask = candidates(frame,base,views)
    y = frame.y.to_numpy(float)
    report = {"note":"saved-model replay and post-screen diagnostics, no fits or new parameter selection",
              "code_sha256":sha(Path(__file__)),"models":[],"raw_clock_parity":{}}
    raw_cols = ["MVT_ID_mvt","MVT_TIME_UTC_mvt","AOBT_3_flt","EOBT_1_flt","LOBT_flt","IOBT_flt"]
    raw = pd.concat([pd.read_parquet(p,columns=raw_cols) for p in paths.values()]).set_index("MVT_ID_mvt",verify_integrity=True).loc[frame.MVT_ID_mvt]
    movement = pd.to_datetime(raw.MVT_TIME_UTC_mvt,utc=True)
    for feature,field in zip(CLOCKS[:4],raw_cols[2:]):
        direct = (movement-pd.to_datetime(raw[field],utc=True)).dt.total_seconds().to_numpy(float)
        cached = frame[feature].to_numpy(float)
        np.testing.assert_array_equal(np.isfinite(direct),np.isfinite(cached))
        valid = np.isfinite(direct)
        difference = float(np.max(np.abs(direct[valid]-cached[valid])))
        if difference>1e-6:
            raise ValueError(f"clock arithmetic mismatch {feature}: {difference}")
        report["raw_clock_parity"][feature] = {"rows":int(valid.sum()),"max_absolute_difference_s":difference}
    for train_month,test_month in ((1,7),(7,1)):
        saved = json.loads((args.run/f"design_{train_month}.json").read_text())
        design = Design()
        design.mean,design.std = np.array(saved["mean"]),np.array(saved["std"])
        design.categories = saved["categories"]
        for family in ("linear","neural"):
            if family=="linear":
                model = np.load(args.run/f"linear_{train_month}.npy")
                predictor = predict_linear
            else:
                dimension = design.transform(frame.iloc[:1],c[:1],mask[:1]).shape[1]
                model = torch.nn.Sequential(torch.nn.Linear(dimension,64),torch.nn.ReLU(),torch.nn.Linear(64,len(PRIORS)-1))
                model.load_state_dict(torch.load(args.run/f"neural_{train_month}.pt",map_location="cpu",weights_only=True))
                model.eval()
                predictor = predict_neural
            item = {"family":family,"train_source_month":train_month,"test_source_month":test_month}
            for label,month in (("training",train_month),("held_out",test_month)):
                rows = source==month
                x = design.transform(frame.loc[rows],c[rows],mask[rows])
                prediction = predictor(model,x,c[rows],mask[rows])
                item[label] = {"rows":int(rows.sum()),
                    "baseline_rmse":float(np.sqrt(np.mean((base[rows]-y[rows])**2))),
                    "selector_rmse":float(np.sqrt(np.mean((prediction-y[rows])**2))),
                    "main_mse_gain":float(np.mean((base[rows]-y[rows])**2-(prediction-y[rows])**2))}
                if label=="held_out":
                    original = pooled.loc[rows,f"{family}_blend1.0"].to_numpy(float)
                    difference = float(np.max(np.abs(prediction-original)))
                    if difference>1e-6:
                        raise ValueError(f"model replay mismatch {family} {train_month}: {difference}")
                    item["replay_max_absolute_difference_s"] = difference
            report["models"].append(item)
    write_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
