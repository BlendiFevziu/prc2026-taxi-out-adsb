# Pre-registration: the stand model K2 trained on more days

Written 30 September 2026, **before any fit**. Follows `PREREG_stand_capacity_retest.md` (K2 passed, v71) and
`PREREG_c1_more_data.md` (C1 on E+B1+B2 passed, v72). NM-free. No upload without the owner.

Arm K2+: K2's recipe (400 x 15, learning rate 0.03, min leaf 80, L2 100, row 0.8, column 0.9, max_bin 63, seeds
177/277/377, 36 inputs incl. detector_taxi_s) trained on the stand rows of E, B1 and B2 together. Reference: K2 as in
v71. Both at weight 1.0, floor 300.

Test: the same 5,862 stand rows of the twenty fresh days in nine cells as the K2 retest, detector rows excluded,
comparator bit-identical. **Disclosed: this block already decided K2 against the old model**; it has never been used to
compare two models trained on E and on E+B1+B2.

Rules, all must hold: pooled gain above zero; above zero without the best 100; January and July above zero; at least
70 % of days above zero; at least two thirds of the cells above zero; every paired seed above zero; planning (17,147
rows) against v72's central 252.91, at least 0.1 s.

A pass leads to v73 = v72 with K2+ in place of K2 on the stand model's rows, built and verified like v71.

## Result, 30 September 2026: pass; v73 built and verified

`k2_more_data.py`, `results/k2_more_data.json`. K2+ trained on 11,464 stand rows. On the 5,862 fresh stand rows:
**+1,852 per row over K2, +413 without the best 100, January +2,478, July +1,188, 9 of 9 cells, every seed, planning
+0.18 s. Pass.** `build_v73.py`, `verify_v73.py`: all 17,147 stand rows reproduced v72 with K2 first; 16,932 change
(RMS 41.6 s); everything else identical to v72. SHA 19e79e35... Prediction fixed in the build: central 252.76, best
252.69, worst 252.86 (from v72's central).
