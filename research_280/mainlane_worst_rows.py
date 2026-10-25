"""What the worst main-lane rows are, and whether timestamp granularity is a usable observable.

Part one decomposes the worst 1 % of main-lane rows, which `mainlane_error_map.py` found carry 34.6 % of main
squared error. Part two follows the lead that came out of it and closes it.

Part two's story, kept because the dead end is instructive. The worst non-copy rows have median `rest` of +1,197 s,
i.e. AOBT_3 twenty minutes later than the written block time, which is what a nominal taxi time would produce if
AOBT_3 were computed as takeoff minus a constant. And `ersatz_s` does carry minute structure: 56.7 % of rows sit
within 5 s of a whole minute against 18.3 % by chance, with lifts of 16.3x at LTFM and 6.0x at LIRF but 1.0x at
LEMD and LSZH. LightGBM cannot build `ersatz_s % 60` out of `ersatz_s`, so this looked like an observable the
model structurally cannot see.

It is not about the anchor. AOBT_3 carries seconds == 0 on 96.6 % of rows, so `ersatz_s % 60` is simply
MVT_TIME's second, and the airport pattern matches exactly: MVT seconds == 0 on 8.1 to 8.5 % of rows at EDDM,
EGLL, LFPG, LIRF and LTFM against 1.6 to 1.8 % at the other five. Within an airport the signal is worth nothing:
removing every per-airport gap in mean signed error perfectly bounds the gain at 0.67 MSE per row, 0.0012 s of
official RMSE, and only 5 of 10 airports agree on its sign between January and July.

By-product worth keeping: `sched_sek` in `main_model.py` MERKMALE is SCHED_TIME's second, which is zero on all
1,787,610 departures of 2025. It is a constant, so the seconds-provenance signal the feature list intends was
never delivered. Replacing it with MVT_TIME's second would deliver it and, per the above, still be worthless.

  python3 -B research_280/mainlane_worst_rows.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
OUTPUT = PROJECT / "research_280/results/mainlane_worst_rows.json"
OFFICIAL = 266.4508


def main():
    frame = pd.read_parquet(BASELINE, columns=["ADEP_mvt", "MVT_TIME_UTC_mvt", "y", "ersatz_s",
                                               "schedule", "gate", "baseline"])
    stamp = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
    frame["second"], frame["month"] = stamp.dt.second, stamp.dt.month
    frame["error"] = frame.baseline - frame.y
    frame["rest"] = frame.y - frame.ersatz_s
    frame["is_copy"] = (frame.y - frame.schedule).abs().le(60)
    sse = float((frame.error ** 2).sum())
    frame["abs_error"] = frame.error.abs()
    k = int(len(frame) * .01)
    worst = frame.nlargest(k, "abs_error")

    print(f"main lane {len(frame):,} rows, RMSE {np.sqrt((frame.error ** 2).mean()):.2f}")
    print(f"worst 1 % = {k:,} rows, {(worst.error ** 2).sum() / sse:.1%} of squared error\n")
    groups, split = {}, {
        "copy the gate missed": worst.is_copy & worst.gate.le(.5),
        "copy the gate caught": worst.is_copy & worst.gate.gt(.5),
        "non-copy, gate fired": ~worst.is_copy & worst.gate.gt(.5),
        "non-copy, ordinary": ~worst.is_copy & worst.gate.le(.5)}
    for name, mask in split.items():
        groups[name] = {"rows": int(mask.sum()), "share_of_main_sse": float((worst.error[mask] ** 2).sum() / sse),
                        "median_abs_error_s": float(worst.error[mask].abs().median()),
                        "median_rest_s": float(worst.rest[mask].median())}
        print(f"  {name:22s} {groups[name]['rows']:5,} rows  {groups[name]['share_of_main_sse']:6.2%} of main SSE  "
              f"median |err| {groups[name]['median_abs_error_s']:5.0f} s  median rest "
              f"{groups[name]['median_rest_s']:6.0f} s")

    rounded = frame.second.eq(0)
    print(f"\nMVT_TIME seconds == 0 on {rounded.mean():.2%} of rows; per-airport mean-error gap:")
    gaps, bound = [], 0.
    for airport, part in frame.groupby("ADEP_mvt", observed=True):
        mask = part.second.eq(0)
        gap = float(part.error[mask].mean() - part.error[~mask].mean())
        jan = float(part.error[mask & part.month.eq(1)].mean() - part.error[~mask & part.month.eq(1)].mean())
        jul = float(part.error[mask & part.month.eq(7)].mean() - part.error[~mask & part.month.eq(7)].mean())
        gaps.append({"airport": str(airport), "rows_rounded": int(mask.sum()), "share_rounded": float(mask.mean()),
                     "gap_s": gap, "gap_january_s": jan, "gap_july_s": jul, "months_agree": np.sign(jan) == np.sign(jul)})
        bound += mask.sum() * gap ** 2
        print(f"  {airport:8s} share {mask.mean():6.2%}  gap {gap:6.1f} s  january {jan:6.1f}  july {jul:6.1f}")
    per_row = bound / len(frame)
    print(f"\nagree on sign: {sum(g['months_agree'] for g in gaps)} of 10")
    print(f"oracle bound if every gap were removed: {per_row:.2f} MSE per row, "
          f"official {OFFICIAL:.4f} -> {np.sqrt(OFFICIAL ** 2 - per_row * .985):.4f}. Closed.")

    OUTPUT.write_text(json.dumps({"note": "saved v47 main-lane January/July holdout, no training",
                                  "worst_one_percent": groups, "seconds_gaps": gaps,
                                  "seconds_oracle_mse_per_row": per_row,
                                  "seconds_oracle_official_rmse": float(np.sqrt(OFFICIAL ** 2 - per_row * .985))},
                                 indent=2, default=bool) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
