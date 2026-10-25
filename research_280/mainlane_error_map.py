"""Where the MAIN lane's error sits per airport, and how concentrated it is.

Rome's fallback lane has been screened to exhaustion (see STATUS.md). Its main lane has not. The six-day NM
subset put LIRF's main-lane baseline RMSE at 459.5 s against 240.5 pooled, which would make Rome about 29 % of
main-lane squared error on 7.9 % of rows. That subset is six days, so this recomputes it on the full saved
January/July holdout of the v47 main ensemble: 339,046 rows, no training, no labels touched beyond the holdout
that already exists.

The question a screen depends on is not which airport is worst but whether its error is concentrated in a few
rows, which the fallback work showed is unattackable, or spread across ordinary flights, which is attackable.
So each airport is reported with the share of its squared error carried by its worst 1 % and worst 20 rows.

  python3 -B research_280/mainlane_error_map.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
OUTPUT = PROJECT / "research_280/results/mainlane_error_map.json"


def main():
    frame = pd.read_parquet(BASELINE, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "y", "baseline"])
    error = (frame.baseline - frame.y).to_numpy(float)
    frame["sq"] = error ** 2
    total = frame.sq.sum()
    print(f"main-lane holdout: {len(frame):,} rows, RMSE {np.sqrt(frame.sq.mean()):.3f}")
    order = np.sort(frame.sq.to_numpy())[::-1]
    for k in (5, 20, 100, int(len(frame) * .01)):
        print(f"  worst {k:>5,} rows carry {order[:k].sum() / total:6.2%} of squared error")

    rows = []
    for airport, part in frame.groupby("ADEP_mvt", observed=True):
        sq = np.sort(part.sq.to_numpy())[::-1]
        one_percent = max(int(len(sq) * .01), 1)
        rows.append({"airport": str(airport), "rows": len(part), "row_share": len(part) / len(frame),
                     "rmse": float(np.sqrt(sq.mean())), "sse_share": float(sq.sum() / total),
                     "worst_20_share_of_own_sse": float(sq[:20].sum() / sq.sum()),
                     "worst_1pct_share_of_own_sse": float(sq[:one_percent].sum() / sq.sum()),
                     "rmse_without_worst_1pct": float(np.sqrt(sq[one_percent:].mean())),
                     "median_abs_error": float(np.median(np.sqrt(part.sq.to_numpy())))})
    table = pd.DataFrame(rows).sort_values("sse_share", ascending=False)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # how much of Rome's excess is the tail rather than the ordinary flights
    rome = frame.loc[frame.ADEP_mvt.eq("LIRF")]
    rest = frame.loc[~frame.ADEP_mvt.eq("LIRF")]
    ordinary = lambda part: part.loc[np.sqrt(part.sq) <= 900]
    print(f"\nLIRF ordinary rows (absolute error <= 900 s): {len(ordinary(rome)):,} of {len(rome):,} "
          f"({len(ordinary(rome)) / len(rome):.1%}), RMSE {np.sqrt(ordinary(rome).sq.mean()):.1f}")
    print(f"rest ordinary rows: {len(ordinary(rest)):,} of {len(rest):,} "
          f"({len(ordinary(rest)) / len(rest):.1%}), RMSE {np.sqrt(ordinary(rest).sq.mean()):.1f}")

    OUTPUT.write_text(json.dumps({"note": "saved v47 main-lane January/July holdout, no training",
                                  "rows": int(len(frame)), "rmse": float(np.sqrt(frame.sq.mean())),
                                  "concentration": {str(k): float(order[:k].sum() / total)
                                                    for k in (5, 20, 100, int(len(frame) * .01))},
                                  "by_airport": table.to_dict("records")}, indent=2) + "\n")
    print(f"\nwritten {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
