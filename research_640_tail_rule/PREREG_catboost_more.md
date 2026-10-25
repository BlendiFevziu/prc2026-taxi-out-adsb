# Pre-registration: CatBoost as second family for the drift-robust models and for the old C1 at Munich and Madrid

Written 1 October 2026, **before any fit**, under `POLICY_tail_rule.md`. Follows `PREREG_catboost_layers.md` (passed for
C1+ and K2+, v78). NM-free. No upload without the owner.

## Decision A: the drift-robust models (v76, v77)

CatBoost (800 iterations, depth 6, learning rate 0.05, l2 10, RMSE, seeds 177/277/377) on exactly the drift-robust
model's inputs (stand: 27; C1: 24) and training rows (E + B1 + B2). Arm max(base + (drift-robust + CatBoost) / 2, 300)
against the drift-robust model alone. Tested where each was tested: stand on LEMD/07 (research_400 July days, 904 rows)
and EGLL/01 (fresh January stand rows, 765); C1 on EGLL/01 (fresh January partial rows, 3,252) and LEMD/07 (2,777).
Admission follows the original: EGLL/07 through EGLL/01, LEMD/01 and LEMD/07 through LEMD/07.

## Decision B: the old C1 at EDDM and LEMD/07 (v64, v65)

CatBoost on C1's 35 inputs and C1's own training rows E. Arm max(base + (C1 + CatBoost) / 2, 300) against C1 alone.
Tested on research_400's EDDM/01 (5,014) and LEMD/07 (2,777) partial rows; EDDM/07 follows EDDM/01 as in v65.

## Rules per tested cell, all must hold

Pooled gain above zero; airport-day bootstrap 95 % lower bound above zero; both registered half-blocks above zero;
every paired seed above zero. Planning over the admitted 2026 rows (EGLL/07 stand 10,242 and partial 9,048, LEMD/07
stand 2,812, LEMD/01 stand 2,176 and partial 12,478, EDDM/01 6,802, EDDM/07 9,539, LEMD/07 C1 8,692) against v78's
central 246.16, at least 0.05 s. Passing cells enter v79.

## Result, 1 October 2026: all six tested cells pass

`catboost_more.py`, `results/catboost_more.json`. A: drift-robust stand LEMD/07 +911 per row [13, 2,278], EGLL/01 +1,990
[867, 3,394]; drift-robust C1 EGLL/01 +2,232 [917, 3,919], LEMD/07 +848 [287, 1,453]. B: old C1 EDDM/01 +458 [211, 790],
LEMD/07 +1,128 [758, 1,481]. Halves and seeds positive everywhere. Planning +0.43 s as computed.
**Correction disclosed:** the EDDM rows listed above (6,802 and 9,539) included the Munich detector's rows; the old C1
corrects 5,397 (EDDM/01) and 7,525 (EDDM/07) base-owned rows (v64, v65). The build uses those; the planning value falls
slightly and stays far above 0.05 s.

## Candidate v79, built and verified (1 October 2026)

`build_v79.py`, `verify_v79.py`: drift-robust stand 15,230 rows, drift-robust C1 21,526, old C1 21,614 (EDDM/01 5,397,
EDDM/07 7,525, LEMD/07 8,692); every row reproduced v78 from its model first; 58,309 change (RMS 20.5 s); everything
else identical to v78; the verifier recomputed every value with each model's own feature names and the base from the
pre-NM file. SHA 3799ee12... Prediction fixed in the build: central 245.87, best 245.70, worst 246.08 (from v78's
central 246.16).
