# Pre-registration: the drift-robust C1 at Paris

Written 30 September 2026, **before any Paris row is scored**. Same model, inputs and application as
`PREREG_drift_robust_c1.md` (all airport flags zero). NM-free, no fit. No upload without the owner.

Targets: LFPG/07 (1,896) and LFPG/01 (1,890, sister month) base-owned partial rows in 2026. Test: the LFPG partial
rows of the 24 July 2025 days of research_380's stage-1 extraction, joined to the held-out components; **never scored
against labels before**. Rules, all must hold: pooled gain over the base above zero; above zero without the best 100;
both halves of the days above zero; every seed above zero. Each target month is admitted only if the drift rule on the
kept summaries flags nothing between the test rows and its 2026 rows. Planning over the admitted months against
v77's central 247.13, at least 0.05 s. A pass leads to v78 = v77 plus the model on the admitted rows.

## Result, 30 September 2026: fail

`drift_robust_c1_paris.py`, `results/drift_robust_c1_paris.json`. 702 LFPG July partial rows: +15,654 per row, halves and
seeds positive, no drift for either 2026 month, but **-10,771 without the best 100**. Paris stays at the base.
