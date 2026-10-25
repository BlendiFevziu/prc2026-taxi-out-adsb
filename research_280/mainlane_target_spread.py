"""How wide is the main lane's target per airport, and how much of it does the model already explain?

Closes the Rome main-lane line. `mainlane_error_map.py` found LIRF at 357.6 s against 216.8 pooled with the
excess spread across ordinary rows, `gate_calibration_by_airport.py` rejected a per-airport gate exponent, and
`copy_rate_signal.py` rejected smoothed stand and operator copy rates. What none of those asked is whether the
target itself is wider at Rome.

It is. The main-lane target is `rest = y - ersatz_s = AOBT_3 - BLOCK_TIME`, the gap between the A-CDM off-block
and the block time the airport writes down. This reports its spread per airport on non-copy rows, next to the
model's error on the same rows, so the two can be compared directly. Saved holdout, no training.

  python3 -B research_280/mainlane_target_spread.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
OUTPUT = PROJECT / "research_280/results/mainlane_target_spread.json"


def main():
    frame = pd.read_parquet(BASELINE, columns=["ADEP_mvt", "y", "ersatz_s", "schedule", "baseline"])
    frame["rest"] = frame.y - frame.ersatz_s
    copy = (frame.y - frame.schedule).abs() <= 60
    rows = []
    for airport, part in frame.groupby("ADEP_mvt", observed=True):
        part = part.loc[~copy.loc[part.index]]
        rest = part.rest.to_numpy(float)
        low, high = np.percentile(rest, [1, 99])
        spread = float(rest[(rest > low) & (rest < high)].std())
        error = float(np.sqrt(np.mean((part.baseline - part.y).to_numpy(float) ** 2)))
        rows.append({"airport": str(airport), "noncopy_rows": int(len(part)),
                     "rest_median_s": float(np.median(rest)),
                     "rest_iqr_s": float(np.percentile(rest, 75) - np.percentile(rest, 25)),
                     "rest_p01_s": float(low), "rest_p99_s": float(high),
                     "rest_trimmed_sd_s": spread, "model_rmse_s": error,
                     "error_over_spread": error / spread})
    table = pd.DataFrame(rows).sort_values("model_rmse_s", ascending=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nThe last column is the point: a high value means the model explains little of the spread it faces.")
    OUTPUT.write_text(json.dumps({"note": "non-copy main-lane rows of the saved v47 January/July holdout; "
                                          "rest = y - ersatz_s = AOBT_3 - BLOCK_TIME; no training",
                                  "by_airport": rows}, indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
