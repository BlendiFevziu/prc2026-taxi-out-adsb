# Pre-registration: a residual corrector on the out-of-fold base, for the rows no ADS-B layer corrects

Written 30 September 2026, **before any fit**. Prompted by two leading teams' public methodology notes (Discord,
27-28 September): "correct a strong base model rather than predicting from scratch" and "a corrector trained on
out-of-fold base preds (folds by month) did more for us than new features". Our correctors (C1, stand model, detectors)
act only where ADS-B tracks exist; 234,650 scored rows (all of LTFM, LFPG and LIRF's main lane, and parts of the
others) carry the base alone. NM-free, challenge data only. No upload without the owner.

## Data

The held-out January and July 2025 main-lane rows of `harness.components_2025()` (339,046; each predicted by base
models trained without its month). Base = `layers.compose_base` (bit-identical to v75's base on 2026 rows by
construction of `harness.current_base_2026`). Target: label minus base.

## Model, fixed now

LightGBM regression, 400 trees, learning rate 0.03, 31 leaves, min_child_samples 200, lambda_l2 100,
feature_fraction 0.8, bagging 0.8, seeds 177/277/377 averaged. Inputs available for every 2026 row: the base value;
MVT - AOBT_3; MVT - SCHED; AOBT_3 - EOBT_1; AOBT_3 - IOBT; AOBT_3 - LOBT; local hour of MVT and weekday; categoricals
ADEP, RUNWAY_mvt, STAND_mvt, AIRCRAFT_TYPE_flt, AIRCRAFT_OPERATOR_flt, MARKET_SEGMENT_flt, FLIGHT_TYPE_flt,
WK_TBL_CAT_flt, destination country (ADES_mvt[:2]). New value max(base + correction, 300).

## Test: January to July and July to January

Fit on one month's rows, score the other's. Rules, all must hold: gain above zero in both directions; above zero
without the best 100 rows in both directions; at least seven of the ten airports above zero (both directions pooled);
planning: per airport, the pooled gain per row times its 2026 base-owned rows (`harness.ownership_v62`, owner "base"),
over 344,841, against v75's official 249.3908, at least 0.2 s.

A pass leads to v76 = v75 with the corrector (refitted on January and July together) on the 2026 base-owned rows of
the airports that are positive; every other row stays.

## Result, 30 September 2026: fail on every rule

`base_corrector.py`, `results/base_corrector.json`. Base RMSE 214.69 on 339,046 held-out rows, corrected 217.67.
Testing on January -926 per row (-1,400 without the best 100), on July -1,587 (-1,915); **all ten airports negative**
(LTFM -3,420, LIRF -1,776). What a corrector learns from one month's out-of-fold residuals does not carry to the other:
the ensemble base is already calibrated and the remaining structure is month-specific. Nothing built.
