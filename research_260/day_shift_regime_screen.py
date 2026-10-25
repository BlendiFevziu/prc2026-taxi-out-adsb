"""Pre-registered rare-regime screen: a day-shift component for late fallback departures, nested by month, with a label-free
2026 expectation.

Pre-specified 14.09.2026 before any evaluation of the rule, after research_260/late_day_shift_analysis.py. No leaderboard feedback.

Regime: a departure delayed past midnight whose airport block time is the actual off-block clock time stamped with the SCHEDULED
date, so TAXITIME is about 86,400 s plus taxi. In 2025 it occurs only in the fallback lane (no NM record), 11 late rows, all Rome,
mostly among departures at or after 12 UTC (7 day shifts, 2 copies) and rarely before (4 against 52 copies). That split was seen in
the descriptive table, so the rule's thresholds are not taken from it: they are re-selected inside every fold.

Rule family (grid fixed now): scope in {LIRF, all airports} x movement UTC hour >= {0, 6, 12, 18} x delay >= {0, 6, 12, 18} h,
always restricted to fallback rows whose movement date differs from the scheduled date. Within the selected subset,
  p   = (day shifts + 1) / (rows + 2) on the training months (Beta(1, 1) posterior mean)
  tau = median implied taxi (y - 86,400) of the training months' day-shift rows (951 s when none)
  new prediction = (1 - p) * v46 + p * (86,400 + tau); rows outside the subset keep v46.
Selection: the grid point with the largest squared-error gain on the training months; no change when none gains.
Evaluation: twelve month folds (select on eleven months, apply to the twelfth) on the six-pair held-out v46 predictions
(research_240/fallback_anchor_production.py, prod_anchor, 22,470 rows). Those predictions come from models trained without each
row's own pair; the other months' predictions may have seen the held-out month, a weak indirect link that is reported, not removed.
2026: the rule selected on all twelve months is applied to the ranking fallback rows, label-free, against the served v46/v47
fallback predictions. Monte Carlo (20,000 draws): p_true ~ Beta(shifts + 1, rows - shifts + 1) from 2025, each changed row a day
shift with probability p_true (y = 86,400 + a 2025 implied taxi drawn at random), otherwise a copy (y = delay) or a normal
departure (y = 900 s) in the 2025 proportions of the selected subset. Leaderboard MSE change = summed squared-error change / 344,841.

Decision rule for proposing a v48 fallback change (nothing is uploaded here):
  (1) held-out squared-error gain positive over the twelve folds, (2) still positive without the single most favourable row,
  (3) 2026 expected leaderboard MSE gain positive and positive in at least 70 % of the draws, (4) at most 20 changed 2026 rows.

Run from the repository root:  python3 -B research_260/day_shift_regime_screen.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ["1_7", "2_8", "3_9", "4_10", "5_11", "6_12"]
SCOPES, HOURS, DELAYS = ["LIRF", "all"], [0, 6, 12, 18], [0, 6, 12, 18]
ROWS_2026, V47, DEFAULT_TAU, DRAWS = 344841, 266.9416, 951.0, 20000
OUTPUT = ROOT / "research_260/results/day_shift_regime_screen.json"


def timing(frame):
    t_mvt, t_sch = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True), pd.to_datetime(frame.SCHED_TIME_UTC_mvt, utc=True)
    frame = frame.copy()
    frame["delay"] = (t_mvt - t_sch).dt.total_seconds().to_numpy()
    frame["late"] = (t_mvt.dt.date != t_sch.dt.date).to_numpy()
    frame["hour"] = t_mvt.dt.hour.to_numpy()
    frame["airport"] = frame.ADEP_mvt.astype(str)
    return frame


pred = pd.concat([pd.read_parquet(ROOT / f".cache/research_240/fallback_anchor_production/predictions_{p}.parquet",
                                  columns=["MVT_ID_mvt", "y", "prod_anchor", "fold"]) for p in PAIRS], ignore_index=True)
parts = []
for path in sorted((ROOT / "data").glob("training_2025-*.parquet")):
    raw = pd.read_parquet(path, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    raw = raw[raw.PHASE_mvt.eq("DEP") & raw.MVT_ID_mvt.isin(pred.MVT_ID_mvt)].copy()
    raw["month"] = int(path.name[14:16])
    parts.append(raw)
d = timing(pd.concat(parts, ignore_index=True).merge(pred, on="MVT_ID_mvt", validate="one_to_one"))
assert len(d) == 22470
y, v46 = d.y.to_numpy(float), d.prod_anchor.to_numpy(float)
copy = np.abs(y - d.delay.to_numpy()) <= 60
shift = ~copy & (y >= 80000) & (y <= 93600)
normal = ~copy & (y >= -600) & (y <= 5400)
d["shift"], d["copy"], d["normal"] = shift, copy, normal


def subset(frame, scope, hour, delay_h):
    mask = frame.late.to_numpy() & (frame.hour.to_numpy() >= hour) & (frame.delay.to_numpy() >= delay_h * 3600)
    if scope == "LIRF":
        mask &= frame.airport.to_numpy() == "LIRF"
    return mask


def fit(train):
    tau = float(np.median(train.y[train["shift"]] - 86400)) if train["shift"].any() else DEFAULT_TAU
    best = None
    for scope in SCOPES:
        for hour in HOURS:
            for delay_h in DELAYS:
                mask = subset(train, scope, hour, delay_h)
                n = int(mask.sum())
                if not n:
                    continue
                s = int((mask & train["shift"].to_numpy()).sum())
                p = (s + 1) / (n + 2)
                ty, tv = train.y.to_numpy(float)[mask], train.prod_anchor.to_numpy(float)[mask]
                gain = float(np.sum((tv - ty) ** 2 - ((1 - p) * tv + p * (86400 + tau) - ty) ** 2))
                if best is None or gain > best["training_gain"]:
                    best = {"scope": scope, "hour": hour, "delay_h": delay_h, "rows": n, "shifts": s, "p": p, "tau": tau, "training_gain": gain,
                            "copies": int((mask & train["copy"].to_numpy()).sum()), "normals": int((mask & train["normal"].to_numpy()).sum())}
    return best if best is not None and best["training_gain"] > 0 else None


new = v46.copy()
folds = {}
for month in range(1, 13):
    held = d.month.to_numpy() == month
    rule = fit(d[~held])
    if rule is None:
        folds[str(month)] = {"rule": None, "changed_rows": 0, "gain": 0.0}
        continue
    mask = held & subset(d, rule["scope"], rule["hour"], rule["delay_h"])
    new[mask] = (1 - rule["p"]) * v46[mask] + rule["p"] * (86400 + rule["tau"])
    gain = float(np.sum((v46[mask] - y[mask]) ** 2 - (new[mask] - y[mask]) ** 2))
    folds[str(month)] = {"rule": {k: rule[k] for k in ("scope", "hour", "delay_h", "p", "tau")}, "changed_rows": int(mask.sum()),
                         "shift_rows_changed": int((mask & shift).sum()), "copy_rows_changed": int((mask & copy).sum()), "gain": gain}
row_gain = (v46 - y) ** 2 - (new - y) ** 2
changed = new != v46
order = np.sort(row_gain[changed])[::-1]
heldout = {"changed_rows": int(changed.sum()), "sse_gain": float(row_gain.sum()), "fallback_mse_gain_per_row": float(row_gain.mean()),
           "v46_fallback_rmse": float(np.sqrt(np.mean((v46 - y) ** 2))), "new_fallback_rmse": float(np.sqrt(np.mean((new - y) ** 2))),
           "sse_gain_without_top1_row": float(row_gain.sum() - (order[0] if len(order) else 0)),
           "sse_gain_without_top2_rows": float(row_gain.sum() - order[:2].sum()),
           "leaderboard_mse_equivalent_at_2026_fallback_share": float(row_gain.mean() * 5290 / ROWS_2026),
           "folds_with_change": int(sum(f["changed_rows"] > 0 for f in folds.values())),
           "folds_positive": int(sum(f["gain"] > 0 for f in folds.values())), "folds_negative": int(sum(f["gain"] < 0 for f in folds.values()))}

final = fit(d)
exposure = {"rule": final}
if final is not None:
    ids26 = pd.read_parquet(next((ROOT / ".cache/research_240/anchor_preview").glob("fallback_2026_ranking_*.parquet")), columns=["MVT_ID_mvt"])
    raw26 = pd.read_parquet(ROOT / "data/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    r = timing(raw26[raw26.PHASE_mvt.eq("DEP") & raw26.MVT_ID_mvt.isin(ids26.MVT_ID_mvt)])
    r = r.merge(pd.read_parquet(ROOT / ".cache/research_240/anchor_preview_2026_production.parquet", columns=["MVT_ID_mvt", "anchored"]),
                on="MVT_ID_mvt", validate="one_to_one")
    assert len(r) == 5290
    mask = subset(r, final["scope"], final["hour"], final["delay_h"])
    served = r.anchored.to_numpy(float)[mask]
    delay26 = r.delay.to_numpy(float)[mask]
    proposed = (1 - final["p"]) * served + final["p"] * (86400 + final["tau"])
    taus = (y[shift] - 86400) if shift.any() else np.array([DEFAULT_TAU])
    non_shift = max(final["rows"] - final["shifts"], 1)
    copy_share = final["copies"] / non_shift
    rng = np.random.default_rng(20260914)
    gains = np.zeros(DRAWS)
    for k in range(DRAWS):
        p_true = rng.beta(final["shifts"] + 1, final["rows"] - final["shifts"] + 1)
        is_shift = rng.random(len(served)) < p_true
        is_copy = rng.random(len(served)) < copy_share
        truth = np.where(is_shift, 86400 + rng.choice(taus, len(served)), np.where(is_copy, delay26, 900.))
        gains[k] = np.sum((served - truth) ** 2 - (proposed - truth) ** 2) / ROWS_2026
    rmse_change = V47 - np.sqrt(np.maximum(V47 ** 2 - gains, 0))
    exposure.update({"changed_rows_2026": int(mask.sum()), "delay_h_of_changed_rows": [round(float(v) / 3600, 2) for v in delay26],
                     "served_h": [round(float(v) / 3600, 2) for v in served], "proposed_h": [round(float(v) / 3600, 2) for v in proposed],
                     "expected_leaderboard_mse_gain": float(gains.mean()), "probability_gain_positive": float(np.mean(gains > 0)),
                     "leaderboard_rmse_change_s_quantiles": {q: float(np.quantile(rmse_change, q)) for q in (.05, .25, .5, .75, .95)},
                     "expected_leaderboard_rmse_change_s": float(rmse_change.mean())})
checks = {"1_heldout_gain_positive": heldout["sse_gain"] > 0, "2_positive_without_top1_row": heldout["sse_gain_without_top1_row"] > 0,
          "3_2026_expected_gain_positive_in_70pct": bool(final is not None and exposure["expected_leaderboard_mse_gain"] > 0
                                                         and exposure["probability_gain_positive"] >= .7),
          "4_at_most_20_changed_2026_rows": bool(final is not None and exposure["changed_rows_2026"] <= 20)}
report = {"note": "rare-regime screen; nested by month on six-pair held-out v46 fallback predictions; 2026 label-free Monte Carlo",
          "regime_rows_2025": {"day_shift": int(shift.sum()), "late_day_shift": int((shift & d.late.to_numpy()).sum())},
          "folds": folds, "heldout": heldout, "exposure_2026": exposure, "checks": checks,
          "decision": {"propose_v48_fallback_change": bool(all(checks.values()))}}
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print("folds:", {m: (f["rule"]["scope"] + f">={f['rule']['hour']}h,d>={f['rule']['delay_h']}h,p={f['rule']['p']:.2f}" if f["rule"] else None,
                     f["changed_rows"], round(f["gain"] / 1e6, 1)) for m, f in folds.items()})
print("held-out:", json.dumps(heldout, indent=1))
print("final rule:", json.dumps(final))
print("2026:", json.dumps({k: v for k, v in exposure.items() if k != "rule"}, indent=1))
print("checks:", checks, "decision:", report["decision"])
