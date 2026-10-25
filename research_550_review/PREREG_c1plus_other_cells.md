# Pre-registration: C1+ in the cells where C1 runs outside its eight evaluated cells

Written 30 September 2026, **before C1+ is scored on any row below**. Follows `PREREG_c1_more_data.md` (passed, v72).
NM-free, no fit (C1+ is saved). No upload without the owner.

## Decision (a): Munich January and Madrid July

Since v64, C1 corrects EDDM/01 (6,802 rows in 2026) and LEMD/07 (8,692) with all airport flags at zero. Test rows: the
frozen extractions of `research_400_cells/cells_confirm.py` (21 January, 24 July local days, partial rows joined to the
held-out components). Arm C1+ against C1, both at weight 1.0. Disclosed: those calendar days overlap C1+'s training
days, but no Munich or Madrid row was ever a training row; the rows were read for C1's own test on 29 September.
Rules, per cell, all must hold: pooled gain above zero; above zero without the best 100; both registered half-blocks
above zero; every seed above zero. Planning over the passing cells, against v72's central 252.91, at least 0.05 s.
Munich July (v65, sister-month transfer, no labelled July extraction) is not touched either way.

## Decision (b): Amsterdam January average

Since v68 the 9,752 EHAM January partial rows carry max(base + (add-on + C1) / 2, 300). Arm: the same with C1+.
Test rows: the fresh January EHAM partial rows of `research_480_weights/weights.py` (ten days, 4,686 rows, read for
v68's test), plus its eleven development days reported. Rules: fresh pooled gain above zero; above zero without the
best 100; at least seven of ten days above zero; every seed above zero; planning (9,752 rows) at least 0.05 s.

A passing decision enters v73 = v72 plus that change, built and verified like v72.

## Result, 30 September 2026: both decisions fail

`c1plus_other_cells.py`, `results/c1plus_other_cells.json`. (a) EDDM/01 5,014 rows +1,276 per row but -435 without the
best 100; LEMD/07 2,777 rows +112, -2,133 without the best 100, first half -4. (b) Amsterdam January fresh 4,686 rows
+1,154 per row, 10 of 10 days, every seed, planning +0.06 s, but **-0.6 without the best 100**: fails by a hair, the
rule is not relaxed (development +2,081, but its days overlap C1+'s training days). v72 stays unchanged.
