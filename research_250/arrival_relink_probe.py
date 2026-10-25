"""Read-only feasibility probe: recover an anchor from the same flight's arrival.

Matching reads only supplied input fields. Departure taxi-time labels are joined
afterwards for diagnostics, never for selecting a match. No model or upload.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--months", default="1,7")
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
months = set(map(int, args.months.split(",")))
paths = [p for p in sorted((args.project / "data").glob("training_2025-*.parquet"))
         if int(re.match(r"training_2025-(\d{2})-01_", p.name)[1]) in months]
columns = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
           "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"]
raw = pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True)
for col in ("FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt"):
    raw[col] = raw[col].fillna("").astype(str).str.upper().str.strip()
phase = raw.PHASE_mvt.astype(str).str.upper()
dep = raw[phase.str.startswith("DEP")].copy()
arr = raw[phase.str.startswith("ARR") & raw.AOBT_3_flt.notna()].copy()
fallback = dep[dep.AOBT_3_flt.isna()]
control = dep[dep.AOBT_3_flt.notna()].sample(n=min(20000, int(dep.AOBT_3_flt.notna().sum())), random_state=250)
query = pd.concat([fallback.assign(group="fallback"), control.assign(group="known_anchor_control")])
query = query[query.FLIGHT_mvt.ne("")]
keys = ["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "AIRCRAFT_TYPE_mvt"]
matches = query.merge(arr[keys + ["MVT_ID_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]],
                      on=keys, how="inner", suffixes=("", "_arrival"))
matches["flight_gap_s"] = (matches.MVT_TIME_UTC_mvt_arrival-matches.MVT_TIME_UTC_mvt).dt.total_seconds()
matches = matches[matches.flight_gap_s.between(10*60, 8*3600)].copy()
counts = matches.groupby("MVT_ID_mvt").size()
matches = matches[matches.MVT_ID_mvt.map(counts).eq(1)].copy()
known = matches[matches.group.eq("known_anchor_control")]
diff = (known.AOBT_3_flt_arrival-known.AOBT_3_flt).dt.total_seconds().abs()
ids_known = known.FLIGHT_ID_mvt.notna() & known.FLIGHT_ID_mvt_arrival.notna()
fb = matches[matches.group.eq("fallback")].copy()
fb["arrival_anchor_prediction"] = (fb.MVT_TIME_UTC_mvt-fb.AOBT_3_flt_arrival).dt.total_seconds()
pred = pd.read_parquet(args.predictions)
assert set(pred.MVT_ID_mvt) == set(fallback.MVT_ID_mvt)
combined = pred.merge(fb[["MVT_ID_mvt", "arrival_anchor_prediction"]], on="MVT_ID_mvt", how="left", validate="one_to_one")
matched = combined.arrival_anchor_prediction.notna()
baseline_error = (combined.v44_fallback-combined.y)**2
trial = combined.v44_fallback.copy()
trial.loc[matched] = combined.loc[matched, "arrival_anchor_prediction"].clip(lower=0)
trial_error = (trial-combined.y)**2
worst = baseline_error.nlargest(10).index
def number(value):
    return None if pd.isna(value) else float(value)
report = {"status": "unsupervised feasibility probe; raw anchor replacement only, no model trained or submitted",
          "source_months": sorted(months), "matching_rule": "exact departure/destination/callsign/aircraft; one arrival candidate 10 min to 8 h later",
          "fallback_rows": len(fallback), "fallback_matches": len(fb),
          "known_anchor_control_rows": len(control), "known_anchor_control_matches": len(known),
          "known_anchor_exact_id_agreement": number((known.loc[ids_known,"FLIGHT_ID_mvt"] == known.loc[ids_known,"FLIGHT_ID_mvt_arrival"]).mean()),
          "known_anchor_difference_seconds_quantiles": {str(q): number(diff.quantile(q)) for q in [0,.5,.95,.99,1]},
          "known_anchor_disagreement_over_60s": int(diff.gt(60).sum()),
          "fallback_top10_error_rows_with_matches": int(matched.loc[worst].sum()),
          "matched_fallback_truth_over_5400": int(combined.loc[matched,"y"].gt(5400).sum()),
          "baseline_fallback_rmse": number(np.sqrt(baseline_error.mean())),
          "raw_anchor_replacement_rmse": number(np.sqrt(trial_error.mean())),
          "matched_rows_baseline_rmse": number(np.sqrt(baseline_error[matched].mean())),
          "matched_rows_raw_anchor_rmse": number(np.sqrt(trial_error[matched].mean())),
          "oracle_max_fallback_mse_gain_on_matched_rows": number(baseline_error[matched].sum()/len(combined)),
          "mse_gain": number(baseline_error.mean()-trial_error.mean())}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps(report, indent=2), flush=True)
