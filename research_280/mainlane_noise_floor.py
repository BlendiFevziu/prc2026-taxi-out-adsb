"""How much of the main lane's target is irreducible? Part one of the capacity study.

The main lane predicts `rest = y - ersatz_s = AOBT_3 - BLOCK_TIME`, the gap between the Network Manager's
off-block clock and the block time the airport writes down. The ensemble reaches 216.77 s on the January/July
holdout. The question this answers is whether that is close to the floor or far from it.

Method: near-replicates. Condition on progressively more of what the model can see and measure the spread of
`rest` WITHIN groups that agree on all of it. No model using only that information can beat the within-group
standard deviation, so each level is a lower bound on achievable RMSE at that conditioning. If the bound stops
falling as conditioning gets richer, the target is mostly noise and the lane is close to exhausted.

Copies are reported separately. On a copied block time `rest` equals the schedule delay by construction, which
is a different quantity with its own enormous spread; the gate handles those rows, so mixing them in would
overstate the floor.

Read the output as bounds, not forecasts: richer conditioning means fewer and more selected groups.

  python3 -B research_280/mainlane_noise_floor.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / ".cache/clock_queue_release_20260912/data"
OUTPUT = PROJECT / "research_280/results/mainlane_noise_floor.json"
ENSEMBLE = 216.773      # v47 main-lane RMSE on this holdout
SINGLE_TREE = 223.731   # the `rest` view as a single v45-recipe LightGBM, from research_240/main_target_views


def within(frame, keys):
    """Pooled within-group standard deviation of `rest`, over groups with at least two rows."""
    group = frame.groupby(keys, observed=True).rest
    size, var = group.size(), group.var(ddof=1)
    ok = (size >= 2) & var.notna()
    dof = (size[ok] - 1)
    pooled = float(np.sqrt((var[ok] * dof).sum() / dof.sum()))
    return {"levels": int(ok.sum()), "rows_in_groups": int(size[ok].sum()),
            "share_of_rows": float(size[ok].sum() / len(frame)), "pooled_within_sd_s": pooled}


def main():
    columns = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt",
               "y", "ersatz_s", "versp_min", "hat_aobt", "stunde"]
    frame = pd.read_parquet(DATA / "months_1_7.parquet", columns=columns)
    frame = frame.loc[frame.hat_aobt.astype(int).eq(1)].copy()
    frame["rest"] = frame.y - frame.ersatz_s
    stamp = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
    frame["tag"] = stamp.dt.date.astype(str)
    frame["is_copy"] = (frame.y - frame.versp_min * 60).abs().le(60)
    for column in ("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"):
        frame[column] = frame[column].astype(str)
    ordinary = frame.loc[~frame.is_copy]
    low, high = np.percentile(ordinary.rest, [1, 99])
    print(f"main-lane holdout {len(frame):,} rows; copies {frame.is_copy.mean():.2%}")
    print(f"rest on non-copy rows: median {ordinary.rest.median():.0f} s, "
          f"sd {ordinary.rest.std():.1f}, trimmed sd {ordinary.rest[(ordinary.rest > low) & (ordinary.rest < high)].std():.1f}\n")

    schemes = [("unconditional", []),
               ("airport", ["ADEP_mvt"]),
               ("airport + stand", ["ADEP_mvt", "STAND_mvt"]),
               ("airport + stand + runway", ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]),
               ("+ aircraft type", ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"]),
               ("+ day", ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "tag"]),
               ("+ day + hour", ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "tag", "stunde"]),
               ("airport + day + hour", ["ADEP_mvt", "tag", "stunde"]),
               ("airport + stand + day + hour", ["ADEP_mvt", "STAND_mvt", "tag", "stunde"])]
    results = []
    print(f"{'conditioning':34s} {'groups':>8s} {'rows':>9s} {'share':>7s} {'within sd':>10s}")
    for name, keys in schemes:
        if not keys:
            entry = {"levels": 1, "rows_in_groups": int(len(ordinary)), "share_of_rows": 1.,
                     "pooled_within_sd_s": float(ordinary.rest.std())}
        else:
            entry = within(ordinary, keys)
        entry["conditioning"] = name
        results.append(entry)
        print(f"{name:34s} {entry['levels']:>8,} {entry['rows_in_groups']:>9,} "
              f"{entry['share_of_rows']:>6.1%} {entry['pooled_within_sd_s']:>10.1f}")

    richest = min(r["pooled_within_sd_s"] for r in results if r["conditioning"] != "unconditional")
    print(f"\nensemble on the same rows: {ENSEMBLE:.2f} s (single-tree `rest` view {SINGLE_TREE:.2f})")
    print(f"tightest bound measured:   {richest:.1f} s")
    print(f"headroom implied:          {ENSEMBLE - richest:+.1f} s, i.e. the ensemble is "
          f"{ENSEMBLE / richest:.2f}x the bound")
    print("\nA bound BELOW the ensemble means learnable structure may remain; a bound ABOVE it means the\n"
          "ensemble is already using information these keys do not capture. Both are possible per level.")
    OUTPUT.write_text(json.dumps({"note": "within-group spread of rest on non-copy main-lane holdout rows; "
                                          "bounds at each conditioning, not forecasts",
                                  "ensemble_rmse": ENSEMBLE, "single_tree_rmse": SINGLE_TREE,
                                  "levels": results}, indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
