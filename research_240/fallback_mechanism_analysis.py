"""Which mechanisms produce the largest fallback errors, and can the date-shift pattern be predicted?

No new fit and no leaderboard feedback. Uses the six-pair held-out fallback predictions of the scored
v46 recipe (prod_anchor) saved by research_240/fallback_rescreen_production.py and its cached tables.

Mechanisms per held-out row: copied (|y - 60*versp| <= 60 s, block time equals the schedule), date shift
(85,000 <= y <= 93,600 s and not copied, a block time carrying the scheduled date with the actual clock
time), normal (y <= 5,400 s), other. Tabulated by whether the movement date differs from the scheduled
date (tag_versch) and by airport, with each cell's share of fallback squared error.

Rule test: for rows with tag_versch = 1, replace the prediction by the mean y of the same airport and
delay band (0-6, 6-12, 12-18, 18-24, 24+ h) in the other five held-out pairs (cells with at least five
rows), and compare with prod_anchor. 2026 exposure: tag_versch = 1 rows among the ranking fallback rows.

Run from the repository root:
  python3 research_240/fallback_mechanism_analysis.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(".cache/research_240/rescreen_production"))
    parser.add_argument("--ranking-tables", type=Path, default=Path(".cache/research_240/anchor_preview"))
    parser.add_argument("--output", type=Path, default=Path("research_240/results/fallback_mechanism_analysis.json"))
    args = parser.parse_args()
    index = {tuple(sorted(json.loads(p.read_text())["source_months"])): p.with_suffix(".parquet") for p in args.cache.glob("rescreen_*.json")}
    frames = []
    for a, b in PAIRS:
        pred = pd.read_parquet(args.cache / f"predictions_{a}_{b}.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "y", "prod_anchor"])
        table = pd.read_parquet(index[(a, b)], columns=["MVT_ID_mvt", "versp_min", "tag_versch", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"])
        frames.append(pred.merge(table, on="MVT_ID_mvt", how="left", validate="one_to_one").assign(fold=f"{a}_{b}"))
    d = pd.concat(frames, ignore_index=True)
    y = d.y.to_numpy(float)
    delay_s = d.versp_min.to_numpy(float) * 60
    copied = np.abs(y - delay_s) <= 60
    shifted = (y >= 85000) & (y <= 93600) & ~copied
    d["mechanism"] = np.where(copied, "copied", np.where(shifted, "date_shift", np.where(y <= 5400, "normal", "other")))
    d["se"] = (d.prod_anchor.to_numpy(float) - y) ** 2
    total = d.se.sum()
    block, sched = pd.to_datetime(d.BLOCK_TIME_UTC_mvt, utc=True), pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True)
    d["block_date_equals_sched_date"] = (block.dt.date == sched.dt.date).to_numpy()
    report = {"note": "six-pair held-out fallback rows of the v46 recipe (prod_anchor); no fitting",
              "rows": int(len(d)), "fallback_rmse": float(np.sqrt(d.se.mean())),
              "by_tag_versch_and_mechanism": {f"{int(t)}|{m}": {"rows": int(len(g)), "se_share": float(g.se.sum() / total)}
                                              for (t, m), g in d.groupby(["tag_versch", "mechanism"])}}
    shifted_rows = d[d.tag_versch == 1]
    report["tag_versch_1_by_airport"] = {
        airport: {"rows": int(len(g)), "date_shift_share": float((g.mechanism == "date_shift").mean()),
                  "copied_share": float((g.mechanism == "copied").mean()), "normal_share": float((g.mechanism == "normal").mean()),
                  "other_share": float((g.mechanism == "other").mean()), "se_share": float(g.se.sum() / total),
                  "block_date_equals_sched_date_share": float(g.block_date_equals_sched_date.mean()),
                  "median_delay_h": float(g.versp_min.median() / 60)}
        for airport, g in shifted_rows.groupby(shifted_rows.ADEP_mvt.astype(str))}
    rome_shift = shifted_rows[(shifted_rows.ADEP_mvt.astype(str) == "LIRF") & (shifted_rows.mechanism == "date_shift")]
    report["lirf_date_shift_median_y_minus_86400_s"] = float(np.median(rome_shift.y - 86400)) if len(rome_shift) else None

    bands = pd.cut(d.versp_min / 60, [0, 6, 12, 18, 24, 1e9], right=False, labels=["0-6h", "6-12h", "12-18h", "18-24h", "24h+"])
    key = (d.ADEP_mvt.astype(str) + "|" + bands.astype(str)).to_numpy()
    cell = (d.tag_versch == 1).to_numpy()
    folds = d.fold.to_numpy()
    rule = d.prod_anchor.to_numpy(float).copy()
    for fold in sorted(set(folds)):
        test, train = cell & (folds == fold), cell & (folds != fold)
        means = pd.Series(y[train]).groupby(key[train]).mean()
        counts = pd.Series(y[train]).groupby(key[train]).size()
        idx = np.flatnonzero(test)[pd.Series(key[test]).map(counts).fillna(0).to_numpy() >= 5]
        rule[idx] = pd.Series(key[idx]).map(means).to_numpy()
    gain = d.se.to_numpy() - (rule - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:100]] = False
    report["conditional_mean_rule_tag_versch_1"] = {
        "rows_replaced": int((rule != d.prod_anchor.to_numpy(float)).sum()), "pooled_mse_gain": float(gain.mean()),
        "mse_gain_without_top100_rows": float(gain[keep].mean()),
        "pair_gains": {f: float(gain[folds == f].mean()) for f in sorted(set(folds))}}
    rank_file = sorted(args.ranking_tables.glob("fallback_2026_ranking_*.parquet"))[0]
    rank = pd.read_parquet(rank_file, columns=["ADEP_mvt", "tag_versch"])
    report["tag_versch_1_rows"] = {"held_out_2025": {k: int(v) for k, v in shifted_rows.ADEP_mvt.astype(str).value_counts().items()},
                                   "ranking_2026": {k: int(v) for k, v in rank[rank.tag_versch == 1].ADEP_mvt.astype(str).value_counts().items()},
                                   "ranking_2026_fallback_rows": int(len(rank))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("rows", "fallback_rmse", "by_tag_versch_and_mechanism", "lirf_date_shift_median_y_minus_86400_s",
                                             "conditional_mean_rule_tag_versch_1", "tag_versch_1_rows")}, indent=2))


if __name__ == "__main__":
    main()
