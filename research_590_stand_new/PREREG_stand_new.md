# Pre-registration: the stand model at Munich and Madrid

Written 30 September 2026, **before any stand row of these airports is scored**. Owner: "lets keep improving rmse,
go all out". NM-free, no fit (K2+ is saved). No upload without the owner.

## Why

The stand layer gained every time it was switched on (v57, v66, v67, v71, v73), and C1 transferred to Munich and
Madrid with all airport flags at zero (v64, v65). In 2026, 5,932 stand-observed rows at these airports carry the base
alone: EDDM January 546 and July 398 (outside the Munich detector), LEMD January 2,176 and July 2,812.

## Arm, fixed now

K2+ (`research_550_review/PREREG_k2_more_data.md`, 400 x 15, trained on E+B1+B2 at the five v56 airports), inputs
through stand_experiment.inputs with all airport flags at zero, detector_taxi_s from `extract.py` (the frozen v56
extraction with the v67 patch, OSM stands). Value max(base + K2+ correction, 300); reference the base.

## Tested cells

EDDM/01 (the 21 local January days of research_400) and LEMD/07 (its 24 July days), stand-observed rows, the Munich
detector's own rows excluded (its deployed configuration replayed: flight_mvt matcher, X-Plane-first stands,
arm_test.guarded), each joined to the held-out 2025 components. Rules per cell, all must hold: pooled gain above
zero; above zero without the best 100 rows; both registered half-blocks of days above zero; every seed above zero;
the v56 drift rule (partial_coverage_2026.drift on the 23 track summaries) between the cell's 2025 stand rows and its
2026 stand rows flags nothing.

## Sister months

EDDM/07 and LEMD/01 have no labelled 2025 tracks here. Each may follow its airport only if that airport's tested
cell passed and the drift rule between the tested cell's 2025 stand rows and the sister month's 2026 stand rows flags
nothing (as v65 did for C1 at Munich July).

Planning over the admitted cells (2026 rows times the tested cell's gain per row), over 344,841, against v75's
official 249.3908, at least 0.05 s. A pass leads to v76 = v75 plus K2+ on the admitted cells' base-owned stand rows.

## Result, 30 September 2026: fail, nothing admitted

`stand_new.py`, `results/stand_new.json`. EDDM/01: 386 rows (Munich detector rows excluded), +5,715 per row, both
halves and every seed positive, no drift, but **-5,310 without the best 100**. LEMD/07: 904 rows, +17,942 per row,
+4,561 without the best 100, both halves and every seed positive, but **the drift rule flags seven summaries**
(ground_points, last_ground_age_s, speed_p90, last_moving_age_s, sampled_distance_m, stand_last_m, stand_median_m)
between July 2025 and July 2026. `build_v76.py` stays unrun.
