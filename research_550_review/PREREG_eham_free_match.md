# Pre-registration: the Amsterdam detector on rows only the stand-parked free match finds

Written 30 September 2026, **before any fresh-day row is scored**. Part of the owner's review. NM-free, no fit. No
upload without the owner.

## Why

The Amsterdam direct detector (v55, guarded, X-Plane-first stands) owns only the rows its flight_mvt callsign matcher
finds; on 25 September 68 % of Amsterdam departures were lost at that matcher. `eham_free_guard_probe.py` (December
2025, labels read in transfer_2025): with the detector's own guards, the stand-parked free match (a unique visit ending
within 300 s of MVT with a guarded onset at the assigned stand, no visit claimed twice) adds 780 main-lane rows to the
582 the callsign matcher finds, at MAE 38.8 s against 34.7 s and 91 % against 93 % within 60 s.

## Arm and reference, fixed now

Rows: Amsterdam main-lane departures not covered by the detector's own matcher and found by the free match, as in the
probe (deployed arm_test.guarded plus tow_guard.check). Arm value: MVT minus the detector's onset, floor 300.
Reference: the current policy value of the row: partial_ground in July: base + C1+; partial_ground in January:
base + (add-on + C1) / 2; observed_at_assigned_stand: base + K2+; every other category: the base. Base = the pre-NM
comparator, bit-identical.

## Test

The twenty fresh days (fresh_july 7, eddf_july_confirm 3, fresh_january 10), Amsterdam. Disclosed: their Amsterdam rows
were read on 29 and 30 September for C1+, K2+ and the stand tests; never for the detector. Rules, all must hold: pooled
gain above zero; above zero without the best 100; January and July above zero; at least 70 % of days above zero;
planning: gain per row times 5,600 (4,188 detector rows in 2026 times the probe's 780 / 582), over 344,841, against
v73's central 252.76, at least 0.1 s.

A pass leads to v74 = v73 plus the detector value on the 2026 rows the free match finds, built from the 62 scored days'
tracks and verified like the others.

## Result, 30 September 2026: fail

`eham_free_match.py`, `results/eham_free_match.json`. 1,622 free-only rows: detector RMSE 184 s against the policy's
118 s; -19,812 per row, -0.64 s planned; both months negative. By the extraction's category: observed_at_assigned_stand
753 rows +4,373 per row, unmatched 255 rows +3,623, **partial_ground 614 rows -59,205**: there the extraction's own
callsign match had linked the flight to another track, and the free match overrode it with a different aircraft at the
stand. The claimed set excluded only the detector matcher's visits, not the extraction's. Restricting to the two
positive groups is post-hoc and worth about 0.08 s: not pursued. `build_v74.py` stays unrun.
