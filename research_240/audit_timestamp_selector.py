"""Post-screen diagnostic, no fits or parameter selection.

Compare the sole screen survivor (ridge 1, strength .2) to its fixed, untrained
softmax prior. Cluster bootstrap airport-days to describe uncertainty. This does
not replace the pre-registered screen or select a new production candidate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from research_240.timestamp_selector import PAIRS, candidate_matrix, metrics, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-project",type=Path,required=True)
    parser.add_argument("--run",type=Path,default=Path(".cache/research_240/timestamp_selector_source_split"))
    parser.add_argument("--output",type=Path,default=Path("research_240/results/timestamp_selector_audit.json"))
    args = parser.parse_args()
    cache = args.cache_project/".cache/research_240/rescreen_production"
    index = {tuple(sorted(json.loads(p.read_text())["source_months"])):p.with_suffix(".parquet") for p in cache.glob("rescreen_*.json")}
    frames = []
    for pair in PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        va = pd.read_parquet(index[pair])
        parts = pd.read_parquet(args.run/f"outer_parts_{key}.parquet")
        d = pd.read_parquet(args.run/f"predictions_{key}.parquet")
        np.testing.assert_array_equal(d.MVT_ID_mvt,va.MVT_ID_mvt)
        c,mask = candidate_matrix(va,parts)
        prior = (weights(np.zeros((1,c.shape[1]-1)),np.ones((len(va),1)),mask)*c).sum(axis=1)
        d["constant_prior"] = d.baseline+.2*(prior-d.baseline)
        frames.append(d)
    d = pd.concat(frames,ignore_index=True)
    survivor = d["ridge1.0_blend0.2"].to_numpy(float)
    y,base,prior = d.y.to_numpy(float),d.baseline.to_numpy(float),d.constant_prior.to_numpy(float)
    gains = np.column_stack([(base-y)**2-(survivor-y)**2,(base-y)**2-(prior-y)**2,(prior-y)**2-(survivor-y)**2])
    day = pd.to_datetime(d.MVT_TIME_UTC_mvt,utc=True).dt.strftime("%Y-%m-%d")
    groups = pd.factorize(d.ADEP_mvt.astype(str)+"|"+day)[0]
    n = np.bincount(groups)
    sums = np.column_stack([np.bincount(groups,weights=gains[:,i]) for i in range(3)])
    rng = np.random.default_rng(271828)
    boot = []
    for _ in range(2000):
        sample = rng.integers(0,len(n),size=len(n))
        boot.append(sums[sample].sum(axis=0)/n[sample].sum())
    ci = np.percentile(boot,[2.5,97.5],axis=0)
    report = {"note":"post-screen diagnostic; no fits, no hyperparameter choice, no leaderboard forecast",
        "constant_prior":metrics(d,prior), "survivor":metrics(d,survivor),
        "learned_increment_mse":float(gains[:,2].mean()),
        "fraction_of_survivor_gain_explained_by_untrained_prior":float(gains[:,1].mean()/gains[:,0].mean()),
        "survivor_minus_baseline_mean_s":float(np.mean(survivor-base)),
        "survivor_minus_prior_mean_s":float(np.mean(survivor-prior)),
        "survivor_minus_prior_max_abs_s":float(np.max(np.abs(survivor-prior))),
        "airport_day_bootstrap":{"clusters":len(n),"resamples":2000,"seed":271828,
            "survivor_gain_95pct_mse":ci[:,0].tolist(),"prior_gain_95pct_mse":ci[:,1].tolist(),
            "learned_increment_95pct_mse":ci[:,2].tolist()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
