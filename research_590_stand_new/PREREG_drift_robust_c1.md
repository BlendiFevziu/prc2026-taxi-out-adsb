# Pre-registration: a drift-robust C1 for Heathrow July and Madrid

Written 30 September 2026, **before any fit of this model**. Follows `PREREG_drift_robust.md` (the same idea for the
stand model passed, v76). NM-free. No upload without the owner.

## Feature set, chosen without labels

C1's 35 inputs minus every summary the v56 drift rule flagged in the label-free comparisons of the target cells'
partial rows: EGLL January 2025 (fresh days) against EGLL July 2026; LEMD July 2025 (research_400 days) against LEMD
January 2026 and against LEMD July 2026. Removed (11): ground_points, first_still_age_s, last_still_age_s,
last_moving_age_s, last_ground_age_s, speed_p90, sampled_distance_m, stand_last_m, stand_median_m,
last_near_stand_age_s, last_still_near_stand_age_s. Kept: the six context columns, the other 12 summaries, the airport
flags and is_july (24).

## Model

C1's recipe (800 x 31, learning rate 0.03, min leaf 80, L2 100, row 0.8, column 0.9, max_bin 63, seeds 177/277/377)
on the 24 inputs, trained on the partial rows of E, B1 and B2 (61,388, as C1+). Value max(base + correction, 300).

## Decisions

1. **EGLL/07** (9,048 base-owned partial rows in 2026) as sister month of EGLL/01, tested on the 3,252 partial rows of
   the ten fresh January days (read before for C1+ and the C1 weight test, never for this model) against the base.
2. **LEMD/01** (12,478 base-owned partial rows) as sister month of LEMD/07, tested on research_400's 2,777 LEMD July
   partial rows against the base.
3. **LEMD/07** (8,692 rows now carrying the old C1, which fails today's drift rule there): replace C1 by this model
   only if, on the same 2,777 rows, it beats C1 pooled and without the best 100.

Per tested cell against the base, all must hold: pooled gain above zero; above zero without the best 100; both
half-blocks of days above zero; every seed above zero. A target cell is admitted only if the drift rule, restricted
to the kept summaries, flags nothing between its test rows and its 2026 rows. Planning over the admitted cells
(against the base for 1 and 2, against C1 for 3), over 344,841, against v76's central 248.22, at least 0.1 s.
A pass leads to v77 = v76 plus the admitted changes.

## Result, 30 September 2026: EGLL/07 and LEMD/01 admitted, LEMD/07 not; v77 built and verified

`drift_robust_c1.py`, `results/drift_robust_c1.json`. 24 inputs. EGLL/01 test (3,252 fresh January partial rows):
+22,359 per row over the base, +9,759 without the best 100, both halves and every seed positive; no drift among the
kept summaries against EGLL July 2026: **EGLL/07 admitted (9,048 rows)**. LEMD/07 test (2,777 rows) against the base:
+8,545, +741 without the best 100, both halves and seeds positive; no drift against LEMD January 2026: **LEMD/01 admitted
(12,478 rows)**. Against the old C1 on the same rows: +140 per row but -2,552 without the best 100: LEMD/07 keeps C1.
Planning +1.81 s. `build_v77.py`, `verify_v77.py`: all gates passed (LEMD extraction equals the audit cache; every row
at the base in v76); 21,526 rows change; the verifier recomputed every value with the models' own feature names.
SHA ce8cc7da... Prediction fixed in the build: central 247.13, best 246.41, worst 248.58 (from v76's central).
