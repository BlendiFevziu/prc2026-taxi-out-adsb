# Pre-registration: the drift-robust stand model at Munich

Written 30 September 2026, **before the drift-robust stand model scores any Munich row**. Follows `PREREG_stand_new.md`
(K2+ at EDDM/01: +5,715 per row but -5,310 without the best 100 on 386 rows) and `PREREG_drift_robust.md` (the
drift-robust stand model, v76). NM-free, no fit. No upload without the owner.

Model and application as `PREREG_drift_robust.md` (27 inputs, all airport flags zero). Test: the same 386 EDDM/01 stand
rows (Munich detector rows excluded), read before for K2+, never for this model. Targets: EDDM/01 (546 base-owned stand
rows in 2026) and EDDM/07 (398, sister month). Rules, all must hold: pooled gain over the base above zero; above zero
without the best 100; both half-blocks above zero; every seed above zero; each target admitted only if the drift rule
on the kept summaries flags nothing; planning against v77's official 246.3837, at least 0.03 s (944 rows at most).
A pass leads to v78 = v77 plus the model on the admitted rows.

## Result, 30 September 2026: fail

`drift_robust_munich.py`, `results/drift_robust_munich.json`. 386 rows: +5,363 per row, halves and seeds positive, no
drift for either month, but **-5,854 without the best 100**. Munich's stand rows stay at the base.
