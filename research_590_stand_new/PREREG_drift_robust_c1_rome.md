# Pre-registration: the drift-robust C1 at Rome January

Written 30 September 2026, **before this model scores any Rome row**. Follows `PREREG_drift_robust_c1.md` (passed for
EGLL/07 and LEMD/01, v77). NM-free, no fit. No upload without the owner.

Target: LIRF/01, 6,767 base-owned partial rows in 2026 (Rome's January tracks are 66 % usable; its July 2026 tracks are
not). Test: the sixteen fresh July 2025 days of `research_380_rome/stage2_c1_transfer.py` (6,796 partial rows), where
the full C1 failed on 29 September (+2,681 per row, -8,706 without the best 100); disclosed: these rows were read then,
never for this model. Model, inputs and application exactly as `PREREG_drift_robust_c1.md` (24 inputs, all airport
flags zero). Rules, all must hold: pooled gain over the base above zero; above zero without the best 100; both halves
(first and last eight days) above zero; every seed above zero; the drift rule on the kept summaries between the test
rows and the LIRF January 2026 rows flags nothing; planning (6,767 rows) against v77's central 247.13, at least 0.1 s.
A pass leads to v78 = v77 plus the model on those rows.

## Result, 30 September 2026: fail

`drift_robust_c1_rome.py`, `results/drift_robust_c1_rome.json`. 6,796 Rome July rows: +2,786 per row, **-9,941 without
the best 100** (as concentrated as the full C1 on 29 September), halves and seeds positive; the drift rule also flags
near_stand_fraction against Rome January 2026. Rome January stays at the base.
