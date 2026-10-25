"""A calibrated correction in the one prediction band where the model is conditionally biased low.

Came out of a competitor's Discord remark that reshaping submission outliers toward the training outliers was
what got them past this score band. Tested in `tail_dispersion.py`, the crude version of that idea fails: a global
stretch above a threshold gains pooled but its gain lives in about 100 rows and flips sign without them, and the
leaderboard sweep of 10.09.2026 independently found factor 1.0 optimal.

The refined version survives the first cut. Regressing y on the prediction inside bands shows the FAR tail is
calibrated (slope 1.013 and 0.956 above 2700 s, both consistent with 1), so its narrowness is just the
conditional mean behaving correctly. One band is not: **1800 to 2700 s, slope 1.097, CI [1.054, 1.140]**, on
9,545 main-lane rows with a mean shortfall of 30 s. That is a conditional bias, not a distributional artefact,
and it touches enough rows to be testable.

The correction is the regression itself: inside the band, replace p with a + b p. Fitted on one month and applied
to the other, never on the rows it scores.

Rules, fixed before the run:
  1. pooled MSE gain positive on the holdout
  2. BOTH directions improve (January-fitted on July, and July-fitted on January)
  3. the gain survives removing its 100 most favourable rows
  4. rows outside the band are untouched by construction, and the fitted slope must exceed 1 in both months
     (otherwise the effect is noise rather than the measured bias)

  python3 -B research_280/band_calibration.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT / ".cache/sequence_release_v47/validate/ensemble_baseline.parquet"
OUTPUT = PROJECT / "research_280/results/band_calibration.json"
LO, HI = 1800.0, 2700.0
V52 = 265.2839


def main():
    f = pd.read_parquet(BASELINE, columns=["MVT_TIME_UTC_mvt", "y", "baseline"])
    y = f.y.to_numpy(float)
    p = f.baseline.to_numpy(float)
    month = pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    jan, jul = month == 1, month == 7
    band = (p >= LO) & (p < HI)
    print(f"{len(f):,} main-lane holdout rows; band {LO:.0f}-{HI:.0f} s holds {band.sum():,} "
          f"({band.mean():.2%}), January {int((band&jan).sum()):,}, July {int((band&jul).sum()):,}")

    out = p.copy()
    fits = {}
    for name, fit_on, apply_to in (("january_fitted", jan, jul), ("july_fitted", jul, jan)):
        rows = band & fit_on
        X = np.column_stack([np.ones(rows.sum()), p[rows]])
        beta, *_ = np.linalg.lstsq(X, y[rows], rcond=None)
        fits[name] = {"intercept": float(beta[0]), "slope": float(beta[1]), "fitted_on_rows": int(rows.sum())}
        target = band & apply_to
        out[target] = beta[0] + beta[1] * p[target]
        print(f"  {name}: a {beta[0]:8.2f}  b {beta[1]:.4f}  applied to {int(target.sum()):,} rows")
    out = np.maximum(out, 300.)

    base_sq = (p - y) ** 2
    sq = (out - y) ** 2
    g = base_sq - sq
    keep = np.ones(len(g), bool)
    keep[np.argsort(g)[::-1][:100]] = False
    checks = {"1_pooled_gain_positive": bool(g.mean() > 0),
              "2_both_directions_improve": bool(g[band & jul].mean() > 0 and g[band & jan].mean() > 0),
              "3_survives_top100_removal": bool(g[keep].mean() > 0),
              "4_slope_above_one_in_both_months": bool(all(v["slope"] > 1 for v in fits.values()))}
    per = g.mean() * 0.98466
    res = {"band": [LO, HI], "band_rows": int(band.sum()), "fits": fits,
           "main_rmse_before": float(np.sqrt(base_sq.mean())), "main_rmse_after": float(np.sqrt(sq.mean())),
           "mse_gain": float(g.mean()), "gain_no_top100": float(g[keep].mean()),
           "january_band_gain": float(g[band & jan].mean()), "july_band_gain": float(g[band & jul].mean()),
           "checks": checks, "passes": all(checks.values()),
           "mechanical_official": float(np.sqrt(max(V52 ** 2 - per, 1)))}
    print(f"\nmain-lane RMSE {res['main_rmse_before']:.4f} -> {res['main_rmse_after']:.4f}")
    print(f"pooled gain {res['mse_gain']:,.2f} MSE ({res['gain_no_top100']:,.2f} without the best 100 rows)")
    print(f"band gain: January {res['january_band_gain']:,.0f}, July {res['july_band_gain']:,.0f}")
    print("\nchecks " + json.dumps(checks))
    print(f"PASSES ALL FOUR: {res['passes']}")
    if res["passes"]:
        print(f"mechanical official: {V52:.4f} -> {res['mechanical_official']:.4f} "
              f"({V52 - res['mechanical_official']:+.3f} s), NOT a forecast")
    OUTPUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"written {OUTPUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
