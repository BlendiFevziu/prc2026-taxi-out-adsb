# Pre-registration: a second model family for C1 (CatBoost), verified by rotation

Written 1 October 2026, **before any fit**. Owner: "look at other past experiments and see if we can find anything to
change or try again". Prior: C1 + CatBoost averaged gained +320 per row on 10,872 fresh July rows (29 September) and
failed only without the best 100 (-178), on one block; a leading team named "a second model family on the same inputs"
as its main gain. NM-free. No upload without the owner.

## Arms, fixed now

For each rotation, both families trained on the same rows (partial rows, C1's 35 inputs, target label minus base):
LightGBM with C1's recipe (800 x 31, as `rotation.py` fits it) and CatBoost as in research_430 (800 iterations, depth 6,
learning rate 0.05, l2_leaf_reg 10, RMSE loss, seeds 177/277/377, 4 threads). Arm: max(base + (LGB + CatBoost) / 2, 300)
against max(base + LGB, 300).
- R1: trained on E + B1 + F, scored on B2.
- R2: trained on E + B2 + F, scored on B1.
Rows scored: partial rows of C1+'s eight cells, detector rows excluded, comparator bit-identical.

Rules, all must hold, in **each** rotation: pooled gain above zero; above zero without the best 100; January and July
above zero; at least two thirds of the cells above zero; every paired seed above zero. Planning: the mean rotation gain
per row times 84,970 (C1+'s rows), over 344,841, against v77's official 246.3837, at least 0.1 s.

A pass leads to v78: CatBoost fitted on E + B1 + B2 (C1+'s own training rows) and averaged with the deployed C1+ on its
84,970 rows.

## Result, 1 October 2026: fail as registered; descriptive bootstrap afterwards

`c1_catboost_rotation.py`, `results/c1_catboost_rotation.json`. R1 (on B2): +187 per row, **-378 without the best 100**;
R2 (on B1): +322, **-127**; both months positive in both; planning +0.13 s. Fail as registered, not relaxed.
Descriptive, computed after the verdict and not a decision: airport-day block bootstrap over both rotations (30,643
rows, 88 airport-days, 5,000 draws): mean +273.5, 95 % interval [+126.6, +434.7], no draw at or below zero. Together
with 29 September (+320 on fresh July, -178 without the best 100) the blend has gained pooled on three independent
blocks and failed the best-100 rule on all three: a tail effect that is consistent across days.
