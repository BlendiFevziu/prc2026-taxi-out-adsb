# Pre-registration: the partial-track correction C1 trained on more days

Written 30 September 2026, **before any fit**. Part of the owner's review of closed and settled lines. NM-free. No
upload without the owner.

## Why

C1 (800 x 31) was trained on E, 14 days (24,093 partial rows). Its gains grew with data and capacity each time they
were tested (v58, v60, `research_320/PREREG_capacity2.md`), and the stand model just gained +2,672 per row from the
same step (`PREREG_stand_capacity_retest.md`). Two 2025 blocks have since served only as test data: B1 (development,
14 days) and B2 (confirm, 8 days). The twenty days of 29 September have never trained anything.

## Arm, fixed now

C1+: the deployed C1 recipe unchanged (`layers.fit("partial", ...)`: 800 x 31, learning rate 0.03, min leaf 80, L2 100,
row 0.8, column 0.9, max_bin 63, seeds 177/277/377, 35 inputs), trained on the partial rows of E, B1 and B2 together.
Reference: the deployed C1. Both at weight 1.0, floor 300.

## Test

The twenty days (fresh_july 7, eddf_july_confirm 3, fresh_january 10), partial rows of C1's eight evaluated cells
(EDDF 01/07, EGLL 01, EHAM 07, LEBL 01/07, LSZH 01/07), Amsterdam detector rows excluded, comparator bit-identical.
These rows were read on 29 September for other questions (C1 at 1.25, the C1+CatBoost blend, the second model),
never for this one; disclosed.

Rules, all must hold: pooled gain above zero; above zero without the best 100; January and July above zero; at least
six of eight cells above zero; at least 70 % of days above zero; every paired seed above zero; planning: gain per row
times C1's 2026 rows in these eight cells (84,970), over 344,841, against v71's central 253.36, at least 0.1 s.

A pass leads to v72 = v71 with C1+ in place of C1 on those rows, built and verified like v71.

## Result, 30 September 2026: pass; v72 built and verified

`c1_more_data.py`, `results/c1_more_data.json`. C1+ trained on 61,388 partial rows (E, B1, B2). On 28,539 fresh rows:
**+1,164 per row, +310 without the best 100, January +1,300, July +1,054, 8 of 8 cells, every seed, planning
+0.57 s. Pass.** `build_v72.py`, `verify_v72.py`: all 84,970 C1 rows reproduced v71 with C1 first; 84,355 change (RMS
39.6 s); everything else identical to v71; the verifier rebuilt the inputs through the frozen v56 module. SHA
e0cdcada... Prediction fixed in the build: central 252.91, best 252.68, worst 253.19 (from v71's central).
