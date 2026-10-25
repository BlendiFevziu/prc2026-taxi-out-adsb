"""Decompose 2025 departure taxi-out times into measured and estimated phases, per airport.

Only competition data; no model and no leaderboard feedback. The data holds two airport timestamps per
departure (BLOCK_TIME off-block, MVT_TIME take-off) and the Network Manager's actual off-block AOBT_3.
There are no pushback, engine-start, holding-point or line-up times, so no phase can be measured
exactly beyond the first split below.

Measured, per flight with AOBT_3 and a label (main lane, all twelve 2025 months):
  total          y = MVT_TIME - BLOCK_TIME                          the competition target
  reporting gap  g = AOBT_3 - BLOCK_TIME = y - (MVT_TIME - AOBT_3)  when the two systems stamp off-block
  NM taxi        p = MVT_TIME - AOBT_3                              pushback to take-off

Estimated, following EUROCONTROL's additional taxi-out time indicator (Edition 01.00, March 2023):
  reference   r = 10th percentile of p for each airport, stand and departure runway, over flights with
                  0 < p <= 120 min, valid when at least 10 flights lie at or below it; otherwise the
                  airport-runway 10th percentile is used and the flight is counted as without a valid
                  stand reference. The reference includes pushback and the take-off roll.
  additional  e = p - r: runway queuing and apron or taxiway inefficiency
  split of r  descriptive only: per airport, weighted least squares over valid groups r_g = a + b * d_g
              with d_g the straight-line stand-to-threshold distance (feature strecke_m); a is the part
              not explained by distance, b * d the distance-driven part. This split is unstable and is
              not a measured pushback or taxi duration.

Sanity check: Spearman correlation of e with runway traffic counts, positive if e behaves like queueing.

Run from the repository root:
  python3 research_240/phase_decomposition.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hat_aobt", "y", "ersatz_s", "versp_min", "strecke_m",
           "qt_bahn_vor_10", "qt_bahn_nach_10", "qt_flh_vor_10"]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return None
    return float(np.corrcoef(pd.Series(a[ok]).rank().to_numpy(), pd.Series(b[ok]).rank().to_numpy())[0, 1])


def quantiles(series, levels):
    return {f"p{int(level * 100)}": float(series.quantile(level)) for level in levels}


def reference_split(frame, groups):
    g = groups.dropna(subset=["distance"])
    if len(g) < 10:
        return {"distance_fit_groups": int(len(g))}
    weights = g.n.to_numpy(float)
    design = np.column_stack([np.ones(len(g)), g.distance.to_numpy(float)])
    coef = np.linalg.lstsq(design * np.sqrt(weights)[:, None], g.ref.to_numpy(float) * np.sqrt(weights), rcond=None)[0]
    resid = g.ref.to_numpy(float) - design @ coef
    total = np.average((g.ref.to_numpy(float) - np.average(g.ref, weights=weights)) ** 2, weights=weights)
    distance_part = coef[1] * frame.strecke_m.to_numpy(float)
    return {"not_distance_part_s": float(coef[0]), "distance_s_per_m": float(coef[1]),
            "distance_fit_r2": float(1 - np.average(resid ** 2, weights=weights) / total), "distance_fit_groups": int(len(g)),
            "distance_part_median_s": float(np.nanmedian(distance_part)) if np.isfinite(distance_part).any() else None}


def summary(frame, split):
    out = {"flights": int(len(frame)),
           "valid_reference_share": float(frame.fine_ok.mean()),
           "total_s": quantiles(frame.y, (.25, .5, .75)), "total_mean_s": float(frame.y.mean()),
           "reporting_gap_s": quantiles(frame.gap, (.25, .5, .75)), "reporting_gap_mean_s": float(frame.gap.mean()),
           "reporting_gap_within_60s_share": float((frame.gap.abs() <= 60).mean()),
           "copied_block_time_share": float(frame.copied.mean()),
           "nm_taxi_s": quantiles(frame.ersatz_s, (.25, .5, .75)), "nm_taxi_mean_s": float(frame.ersatz_s.mean()),
           "reference_s": quantiles(frame.ref, (.25, .5, .75)),
           "additional_s": quantiles(frame.excess, (.25, .5, .75, .9)), "additional_mean_s": float(frame.excess.mean()),
           "additional_spearman_runway_departures_10min_before": spearman(frame.excess, frame.qt_bahn_vor_10),
           "additional_spearman_airport_movements_10min_before": spearman(frame.excess, frame.qt_flh_vor_10)}
    out.update(split)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path(".cache/clock_queue_release_20260912"))
    parser.add_argument("--percentile", type=float, default=10.0)
    parser.add_argument("--min-below", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("research_240/results/phase_decomposition.json"))
    args = parser.parse_args()
    table = pd.read_parquet(args.release / "data/months_1_2_3_4_5_6_7_8_9_10_11_12.parquet", columns=COLUMNS)
    all_main = table[table.hat_aobt.eq(1) & table.y.notna() & table.ersatz_s.notna()]
    d = all_main[(all_main.ersatz_s > 0) & (all_main.ersatz_s <= 7200)].copy()
    d["airport"], d["stand"], d["runway"] = d.ADEP_mvt.astype(str), d.STAND_mvt.astype(str), d.RUNWAY_mvt.astype(str)
    d["gap"] = d.y - d.ersatz_s
    d["copied"] = (d.y - d.versp_min * 60).abs() <= 10
    q = args.percentile / 100
    fine, coarse = ["airport", "stand", "runway"], ["airport", "runway"]
    ref_fine = d.groupby(fine).ersatz_s.quantile(q).rename("ref_fine")
    d = d.join(ref_fine, on=fine)
    below = d[d.ersatz_s <= d.ref_fine].groupby(fine).size().rename("n_below")
    d = d.join(below, on=fine).join(d.groupby(coarse).ersatz_s.quantile(q).rename("ref_coarse"), on=coarse)
    d["fine_ok"] = d.n_below.fillna(0).to_numpy() >= args.min_below
    d["ref"] = np.where(d.fine_ok, d.ref_fine, d.ref_coarse)
    d["excess"] = d.ersatz_s - d.ref
    groups = (d[d.fine_ok].groupby(fine)
              .agg(n=("ref_fine", "size"), ref=("ref_fine", "first"), distance=("strecke_m", "median")).reset_index())

    report = {"note": "2025 main-lane departures with AOBT_3 and 0 < MVT - AOBT_3 <= 120 min. Measured: total = reporting gap "
                      "+ NM taxi. Estimated: reference (EUROCONTROL-style 10th percentile per airport, stand and runway) and "
                      "additional time. Not measured phase durations.",
              "percentile": args.percentile, "min_flights_at_or_below_reference": args.min_below,
              "main_lane_flights": int(len(all_main)), "flights_used": int(len(d)),
              "all_airports": summary(d, reference_split(d, groups)), "airports": {}}
    for airport, frame in d.groupby("airport"):
        report["airports"][airport] = summary(frame, reference_split(frame, groups[groups.airport == airport]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print(f"flights used {report['flights_used']:,} of {report['main_lane_flights']:,}; medians in seconds; "
          f"reference = P{args.percentile:.0f} per airport-stand-runway with at least {args.min_below} flights at or below it")
    print(f"{'airport':8s}{'flights':>9}{'valid ref':>10}{'total':>7}{'gap':>6}{'NM taxi':>8}{'ref':>6}{'add.':>6}{'add. p90':>9}{'rho queue':>10}")
    for name, s in [("ALL", report["all_airports"])] + sorted(report["airports"].items()):
        rho = s["additional_spearman_runway_departures_10min_before"]
        print(f"{name:8s}{s['flights']:>9,}{s['valid_reference_share']:>10.1%}{s['total_s']['p50']:>7.0f}{s['reporting_gap_s']['p50']:>6.0f}"
              f"{s['nm_taxi_s']['p50']:>8.0f}{s['reference_s']['p50']:>6.0f}{s['additional_s']['p50']:>6.0f}{s['additional_s']['p90']:>9.0f}"
              f"{('-' if rho is None else format(rho, '.2f')):>10}")


if __name__ == "__main__":
    main()
