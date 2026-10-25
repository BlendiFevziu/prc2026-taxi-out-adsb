# Pre-registration: more training data, round two, verified by rotation

Written 30 September 2026, **before any fit**. Owner: "lets look into this kind of work". NM-free. No upload without
the owner.

## Why

C1 and the stand model gained each time they were given more days (C1+ +1,164 and K2+ +1,852 per row on fresh days).
The deployed C1+, K2+ and both drift-robust models are trained on E, B1 and B2 (36 days). The twenty fresh days (F:
34,679 partial and 6,461 stand rows) have only ever been test data. Deploying on all 56 days cannot be tested on
untouched days any more, so the value of adding F is measured by rotation instead.

## Rotation, fixed now

For each family (C1 recipe: 800 x 31 on the 35 partial inputs; stand recipe: 400 x 15 on the 36 stand inputs; all other
settings as deployed, seeds 177/277/377), two rotations:
- R1: trained on E + B1, against E + B1 + F; both scored on B2.
- R2: trained on E + B2, against E + B2 + F; both scored on B1.
Rows scored: C1 family, partial rows of C1+'s eight cells; stand family, stand rows of K2+'s nine cells; detector rows
excluded; comparator bit-identical. Gain = error without F minus error with F, both at weight 1.0, floor 300.

Rules per family, in **each** rotation: pooled gain above zero; above zero without the best 100; January and July above
zero; at least two thirds of the cells above zero; every paired seed above zero. Planning: half the mean rotation gain
per row (adding F to 36 days will help less than adding it to 22 to 28) times the deployed rows (C1 family 106,496:
C1+ 84,970 and the drift-robust C1 21,526; stand family 32,377: K2+ 17,147 and the drift-robust stand model 15,230),
over 344,841, against v77's official 246.3837, at least 0.1 s.

A passing family is refitted on E + B1 + B2 + F in both its variants (full and drift-robust, same inputs as deployed)
and enters v78 on exactly the rows its current models correct.

## Result, 30 September 2026: both families fail

`rotation.py`, `results/rotation.json`. Gain from adding F, per row: C1 family R1 (on B2) +1,016 but **-156 without the
best 100**; R2 (on B1) +762, +70, pass. Stand family R1 +972, **-493**; R2 +663, **-294**. Pooled gains are positive in
every rotation, but three of four rest on the best 100 rows: at 36 days the layers are near the point where more days
help only a few rows. Planning would have been +0.28 s (C1) and +0.08 s (stand). Nothing refitted; v77 unchanged.
