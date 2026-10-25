"""Label-free sensitivity of v45's scored fallback to the legacy auslast scale skew.

Reads the release run's cached ranking table and final model, reimplements only the
fallback branch of research_250.release_common.predict, and first requires it to reproduce
v45's official fallback predictions. It then gives every row the legacy count an equivalent
row had in the twelve-month 2025 training table (same calendar position, 365 days earlier):

    auslast_cf = auslast * available_days(2025 position, twelve months) / available_days(ranking)

and reports how the fallback predictions move. No 2026 label exists or is read, no score is
used, nothing is written to a bundle, and research_250 and the release stay unchanged.
A prediction change is not an error change; its effect on RMSE is unknown without labels.
"""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--run", default=Path(".cache/clock_queue_release_20260912"), type=Path)
parser.add_argument("--tolerance", type=float, default=0.05)
args = parser.parse_args()
project, output = args.project.resolve(), args.output.resolve()
run = args.run if args.run.is_absolute() else project / args.run
sys.path.insert(0, str(project))
os.chdir(project)

import main_model as H  # torch before tree libraries on macOS; needed to unpickle the bundle
import features_stand_rate as SR
import joblib
import numpy as np
import pandas as pd
from research_240 import features_auslast_rate as AR

table = pd.read_parquet(run / "data/ranking.parquet")
spans_ranking = AR.spans_from_times(table.MVT_TIME_UTC_mvt)
model = joblib.load(run / "final/model.pkl")
official = pd.read_parquet(run / "candidate.parquet").set_index("MVT_ID_mvt", verify_integrity=True)


def fallback_predict(frame):
    """Fallback branch of release_common.predict, unchanged."""
    frame = SR.wende_an(model["standquote"], frame)
    hat = frame.hat_aobt.eq(1).to_numpy()
    x = frame.loc[~hat, model["merkmale_rueck"]]
    q = model["ohne_aobt_klass"].predict_proba(x)[:, 1] ** model.get("hoch", 1.)
    weight = model.get("anteil_hm", 0.)
    high = (1-weight)*frame.loc[~hat, "versp_min"].to_numpy()*60
    if weight:
        high += weight*model["ohne_aobt_hoehe"].predict(x)
    prediction = np.clip((1-q)*model["ohne_aobt"].predict(x) + q*high, 0, None)
    return frame.loc[~hat, "MVT_ID_mvt"].to_numpy(), prediction, q


ids, base, q = fallback_predict(table)
scored = official.loc[ids, "TAXITIME_SEC_mvt"].to_numpy(float)
gate = {"rows": int(len(ids)), "max_abs_difference_s": float(np.abs(base-scored).max()),
        "tolerance_s": args.tolerance, "ranking_spans": [[str(pd.Timestamp(s, tz="UTC")), str(pd.Timestamp(e, tz="UTC"))]
                                                         for s, e in spans_ranking]}
gate["passed"] = gate["max_abs_difference_s"] <= args.tolerance
print("GATE", json.dumps(gate), flush=True)
if not gate["passed"]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"replication_gate": gate}, indent=2))
    sys.exit("Reimplemented fallback branch does not reproduce v45's official fallback predictions; stop.")

counterfactual = table.copy()
t = pd.to_datetime(counterfactual.MVT_TIME_UTC_mvt, utc=True)
days_ranking = AR.available_days(t, spans_ranking)
days_training = AR.available_days(t - pd.Timedelta(days=365), AR.month_spans([(2025, m) for m in range(1, 13)]))
factor = days_training / days_ranking
counterfactual["auslast"] = counterfactual.auslast.to_numpy(float) * factor
ids_cf, moved, q_cf = fallback_predict(counterfactual)
assert np.array_equal(ids_cf, ids)
delta = moved - base

lookup = table.set_index("MVT_ID_mvt", verify_integrity=True).loc[ids]
airports = lookup.ADEP_mvt.astype(str).to_numpy()
when = pd.to_datetime(lookup.MVT_TIME_UTC_mvt, utc=True)
fb_factor = pd.Series(factor, index=table.MVT_ID_mvt.to_numpy()).loc[ids].to_numpy()


def summary(mask):
    d = delta[mask]
    if not d.size:
        return {"rows": 0}
    return {"rows": int(d.size), "mean_abs_delta_s": float(np.abs(d).mean()), "median_delta_s": float(np.median(d)),
            "mean_delta_s": float(d.mean()), "rows_over_5_min": int((np.abs(d) > 300).sum()),
            "rows_over_15_min": int((np.abs(d) > 900).sum()), "rows_over_30_min": int((np.abs(d) > 1800).sum()),
            "max_abs_delta_s": float(np.abs(d).max()),
            "mean_abs_q_change": float(np.abs(q_cf[mask]-q[mask]).mean())}


cluster = (airports == "EHAM") & when.dt.strftime("%Y-%m-%d").between("2026-01-02", "2026-01-06").to_numpy()
report = {"note": "label-free; prediction changes of the scored v45 fallback when legacy auslast is put on its "
                  "twelve-month training scale; not an error or RMSE change",
          "replication_gate": gate,
          "factor_training_over_ranking": {"median": float(np.median(fb_factor)), "p10": float(np.quantile(fb_factor, .1)),
                                           "p90": float(np.quantile(fb_factor, .9))},
          "all_fallback_rows": summary(np.ones(len(ids), bool)),
          "extreme_weight_q_ge_0_1": summary(q >= .1),
          "schiphol_2_to_6_january": summary(cluster),
          "by_month": {int(m): summary((when.dt.month == m).to_numpy()) for m in sorted(when.dt.month.unique())},
          "by_airport": {a: summary(airports == a) for a in sorted(set(airports))}}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ("replication_gate", "factor_training_over_ranking", "all_fallback_rows",
                                         "extreme_weight_q_ge_0_1", "schiphol_2_to_6_january", "by_month")}, indent=2))
print("by airport: " + ", ".join(f"{a} mean|d| {s['mean_abs_delta_s']:.0f}s, >15min {s['rows_over_15_min']}"
                                  for a, s in report["by_airport"].items() if s.get("rows")), flush=True)
