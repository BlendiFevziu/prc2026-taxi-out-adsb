# Pre-registration: hedge Rome's midnight-crossing fallback rows the model calls normal

Written 30 September 2026, **before the leave-one-month-out test below is run**. Owner's request of the same night
("start the audit of the Rome midnight rows"). NM-free, no fit, no upload.

## What was seen first (disclosed)

`audit_crossing.py` and an in-sample crosstab over all 2025 Rome fallback rows. Where the shipped fallback commits to
a normal taxi (value below half the delay) and the delay is 2 to 8 h, the label was a copy on 10 of 22 rows when the
pushback crosses local midnight, against 50 of 211 without a crossing. The model has no crossing input. A pure band-mix
predictor that ignores the model lost to it over 2025 (1.46e10 against 1.37e10), so the model's own discrimination is
kept and only this cell is hedged.

## Rule, fixed now

Rows: Rome (LIRF) departures without AOBT_3, crossing >= 1 (rule.py), delay above 1 h and at most 8 h, and the
shipped value below half the delay. New value: shipped + p x (delay - shipped), where p is the copy share among the
same cell's rows in the months used for estimation. For 2026, p comes from all twelve 2025 months.

## Test: leave one month out over 2025

For each month m, p from the other eleven months' rows of the cell; applied to month m's rows of the cell against the
shipped six-pair out-of-sample value (`prod_anchor`). All must hold:

1. Pooled squared-error gain over all touched 2025 rows above zero.
2. Above zero after removing the 5 most favourable rows (the cell has about 30 rows; 100 would be degenerate).
3. At least two thirds of the months with touched rows above zero.
4. Planning: the pooled gain per touched row times the 2026 rows of the cell, over 344,841, against v69's central
   255.32, at least 0.1 s.

A pass leads to v70 = v69 plus the rule, built and verified like v69. A fail leaves v69 as it is.

## Result, 30 September 2026: fail

`crossing_test.py`, `results/crossing_test.json`. 33 cell rows in 2025 (16 copies, 16 normal, 1 +24 h) over six months.
Pooled gain +6.4e8, but it is June alone (+7.1e8); **without the best 5 rows -2.1e8**; 3 of 6 months positive. The
planning figure (1.55 s on 14 rows in 2026) rests on those few rows and does not count. **Fail**: v69 stays as it is.
With about 45 % copies the cell is a coin flip, and a hedge on a coin flip only pays where the rows are large.
