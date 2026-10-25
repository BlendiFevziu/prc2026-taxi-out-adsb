# Pre-registration: a drift-robust stand model for Madrid and Heathrow July

Written 30 September 2026, **before any fit of this model**. Follows `PREREG_stand_new.md` (LEMD/07 gained +17,942 per
row but failed the drift rule) and a label-free drift check of Heathrow (EGLL July 2026 stand rows are far denser than
any 2025 Heathrow stand rows: ground points median 107 against 44 to 46). NM-free. No upload without the owner.

## Feature set, chosen without labels

K2+'s 36 inputs minus every track summary the v56 drift rule (`partial_coverage_2026.drift`) flagged in the label-free
comparisons of the target cells: LEMD July 2025 against July 2026, and EGLL January 2025 (three blocks) against EGLL
July 2026. Removed (9): ground_points, last_ground_age_s, last_still_age_s, last_moving_age_s, speed_p90,
sampled_distance_m, stand_last_m, stand_median_m, near_stand_fraction. Kept: the six context columns, the other 14
summaries, detector_taxi_s, the airport flags and is_july (27).

## Model

K2+'s recipe (400 x 15, learning rate 0.03, min leaf 80, L2 100, row 0.8, column 0.9, max_bin 63, seeds 177/277/377)
on the 27 inputs, trained on the stand rows of E, B1 and B2 (11,464). Value max(base + correction, 300); reference base.

## Cells and rules

- **LEMD/07**: tested on the 904 stand rows of research_400's 24 July days (airport flags at zero).
- **EGLL/07** (10,242 base-owned stand rows in 2026), as sister month of **EGLL/01**: tested on the 765 stand rows of the
  ten fresh January days, which never trained the model.
- **LEMD/01** (2,176 rows), as sister month of LEMD/07.

Per tested cell, all must hold: pooled gain over the base above zero; above zero without the best 100; both
half-blocks of days above zero; every seed above zero. A cell (or sister month) is then admitted only if the drift
rule, restricted to the kept summaries, flags nothing between its 2025 test rows and its 2026 target rows. Planning
over the admitted cells (tested gain per row times 2026 rows), over 344,841, against v75's official 249.3908, at
least 0.1 s. A pass leads to v76 = v75 plus this model on the admitted cells' base-owned stand rows.

## Result, 30 September 2026: pass; v76 built and verified

`drift_robust.py`, `results/drift_robust.json`. 27 inputs. LEMD/07: 904 rows, +17,287 per row over the base, +3,505
without the best 100, both halves and every seed positive, no drift among the kept summaries (LEMD/07 and LEMD/01 in
2026 admitted). EGLL/01 (sister of EGLL/07): 765 fresh January stand rows (read before for the K2 and K2+ comparisons,
never against the base with this model), +24,205 per row, +5,721 without the best 100, both halves and every seed
positive; no drift among the kept summaries against EGLL July 2026 (admitted). Planning +1.95 s.
`build_v76.py`, `verify_v76.py`: context recipe reproduced Codex's rows; the extraction reproduced the reference on
every target row (EGLL against Codex's context table, LEMD against the coverage-audit cache); all 15,230 rows sat at the
base in v75; all change (mean -41 s, RMS 138 s); the verifier recomputed every value with the models' own feature
names. SHA 4c7afb2d... Prediction fixed in the build: central 248.22, best 247.44, worst 249.78 (central 0.6 of full
transfer because EGLL/07 rests on a sister month with far denser tracks).
