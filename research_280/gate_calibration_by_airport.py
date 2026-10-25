"""Is the main lane's copied-block-time gate calibrated per airport, or only globally?

The gate is one global exponent. main_model.py ERF_HOCH: the prediction on main-lane rows is
`(1 - p**2.5) * model + p**2.5 * schedule_delay`, where p is the fabrication classifier's probability. The
exponent was tuned on one half of a holdout and measured on the other, globally, and the note records an oracle
of 211.58 against 225.32 achieved.

The copy rate is not global. Rome copies the schedule into the block time far more than the other nine airports
(STATUS.md: 48.5 % of its fallback rows, arrival copy rate 12.1 % against 4 to 5.5 %), and research_280's
mainlane_error_map.py puts LIRF's main-lane RMSE at 357.6 against 216.8 pooled, with the excess spread across
ordinary rows rather than in a tail. A single exponent applied to an airport with a several-fold different copy
rate is miscalibrated by construction, whichever way it errs.

This measures that, and nothing else. No training: the saved v47 January/July holdout already carries the gate,
the schedule value and both residual components, and baseline reconstruction from them is exact to 0.000000 s.

Selection discipline, because the exponent is being chosen on the same holdout it is scored on: every per-airport
exponent is chosen on ONE month and applied to the OTHER, and only the cross-applied number is reported as the
result. The in-sample number is printed alongside as an upper bound, clearly labelled.

  python3 -B research_280/gate_calibration_by_airport.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
OUTPUT = PROJECT / "research_280/results/gate_calibration_by_airport.json"
ERF_HOCH = 2.5
GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
NET = 0.25


def apply(normal, schedule, p_raw, exponent):
    gate = p_raw ** exponent
    return np.maximum((1 - gate) * normal + gate * schedule, 300.)


def main():
    frame = pd.read_parquet(BASELINE)
    blend = (1 - NET) * frame.non_neural_residual + NET * frame.old_neural_residual
    normal = (frame.ersatz_s + blend).to_numpy(float)
    schedule = frame.schedule.to_numpy(float)
    y = frame.y.to_numpy(float)
    gate = frame.gate.to_numpy(float)
    p_raw = gate ** (1 / ERF_HOCH)
    base = apply(normal, schedule, p_raw, ERF_HOCH)
    assert np.abs(base - frame.baseline.to_numpy(float)).max() < 1e-9, "baseline reconstruction failed"
    copy = np.abs(y - schedule) <= 60
    airport = frame.ADEP_mvt.astype(str).to_numpy()
    month = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    jan, jul = month == 1, month == 7
    print(f"{len(frame):,} main-lane holdout rows, pooled RMSE {np.sqrt(np.mean((base - y) ** 2)):.3f}, "
          f"copy rate {copy.mean():.2%}, mean gate {gate.mean():.4f}")

    rows, cross = [], base.copy()
    for code in sorted(set(airport)):
        mask = airport == code
        best = {}
        for name, choose, apply_to in (("jan_on_jul", jan & mask, jul & mask), ("jul_on_jan", jul & mask, jan & mask)):
            sse = {h: float(np.mean((apply(normal[choose], schedule[choose], p_raw[choose], h) - y[choose]) ** 2))
                   for h in GRID}
            best[name] = min(GRID, key=sse.get)
            cross[apply_to] = apply(normal[apply_to], schedule[apply_to], p_raw[apply_to], best[name])
        in_sample = {h: float(np.mean((apply(normal[mask], schedule[mask], p_raw[mask], h) - y[mask]) ** 2))
                     for h in GRID}
        best_in = min(GRID, key=in_sample.get)
        oracle = np.where(copy[mask], schedule[mask], normal[mask])
        rows.append({"airport": code, "rows": int(mask.sum()), "copy_rate": float(copy[mask].mean()),
                     "mean_gate": float(gate[mask].mean()),
                     "rmse_now": float(np.sqrt(in_sample[ERF_HOCH])),
                     "best_exponent_in_sample": best_in,
                     "rmse_best_in_sample": float(np.sqrt(in_sample[best_in])),
                     "exponent_from_january": best["jul_on_jan"], "exponent_from_july": best["jan_on_jul"],
                     "rmse_cross_applied": float(np.sqrt(np.mean((cross[mask] - y[mask]) ** 2))),
                     "rmse_perfect_gate_oracle": float(np.sqrt(np.mean((np.maximum(oracle, 300.) - y[mask]) ** 2)))})
    table = pd.DataFrame(rows).sort_values("rows", ascending=False)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    gain = (base - y) ** 2 - (cross - y) ** 2
    keep = np.ones(len(gain), bool)
    keep[np.argsort(gain)[::-1][:100]] = False
    result = {"pooled_rmse_now": float(np.sqrt(np.mean((base - y) ** 2))),
              "pooled_rmse_cross_applied": float(np.sqrt(np.mean((cross - y) ** 2))),
              "pooled_mse_gain": float(gain.mean()),
              "pooled_mse_gain_without_best_100": float(gain[keep].mean()),
              "january_mse_gain": float(gain[jan].mean()), "july_mse_gain": float(gain[jul].mean()),
              "airports_improving": int(sum(r["rmse_cross_applied"] < r["rmse_now"] for r in rows)),
              "copy_rate_overall": float(copy.mean())}
    print("\ncross-applied, the honest number: " + json.dumps({k: round(v, 4) for k, v in result.items()}))
    OUTPUT.write_text(json.dumps({"note": "exponent chosen on one month, applied to the other; in-sample column "
                                          "is an upper bound, not a result. No training.",
                                  "grid": GRID, "by_airport": rows, "pooled": result}, indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
