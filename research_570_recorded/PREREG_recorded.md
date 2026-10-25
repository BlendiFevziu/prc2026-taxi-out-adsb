# Pre-registration: +24 h at Rome as "a time was recorded", learned from every late row

Written 30 September 2026, **before any fit**. Owner's question of the same evening ("how are the others so far
ahead, what could help us solve these rows"). NM-free, challenge data only. No upload without the owner.

## Why

At our level one Rome row missed by 11 h costs about 9.3 s. In 2026, 12 Rome fallback rows pushed back after local
midnight with 8 to 31 h of delay; each is either a copy (label = delay) or a +24 h stamp (label = taxi + 86,400 s).
`research_510` treated +24 h as its own class (12 training rows) and broke in that band. The mechanism
(`research_500_midnight`) says: a copy means no pushback time was recorded; +24 h means a time WAS recorded and the
pushback fell after local midnight. For a crossing row with more than 8 h of delay the pushback is after midnight, so
P(+24 h) = P(recorded). "Recorded" (normal or +24 h) against "copy" is learnable from all 1,363 late Rome fallback rows.

## Model, fixed now

LightGBM binary, recorded (label not within 60 s of the delay) against copy, on Rome fallback departures with delay
above 1 h: the research_510 inputs and settings (log delay, crossing, local hours of SCHED and MVT, weekday, airline,
destination country, stand area, runway; 200 trees, learning rate 0.03, 7 leaves, min_child_samples 20, lambda_l2 10,
feature_fraction 0.9, bagging 0.8, seeds 177/277/377). Expected value on a target row: p x (86,400 + t) +
(1 - p) x delay, with t the median taxi of the training months' normal rows.

Target rows: Rome fallback departures with crossing >= 1 and delay above 8 h. Arm A (decision): the average of the
shipped value and the expected value. Arm B (reported): the expected value alone.

## Test: leave one month out over 2025

For each month, the model is fitted on every other month's late Rome fallback rows and scores that month's target
rows; reference the shipped six-pair value (`prod_anchor`). About 38 target rows (27 copy, 11 +24 h). Rules, all must
hold for arm A: pooled gain above zero; above zero without the best 3 rows; AUC of p for +24 h against copy on the
target rows at least 0.65; at least two thirds of the months with target rows above zero; planning: gain per target
row times the 2026 target rows, over 344,841, against v75's official 249.3908, at least 0.1 s.

A pass leads to v76 = v75 plus arm A on the 2026 target rows, the model refitted on all twelve months.

## Result, 30 September 2026: fail

`recorded.py`, `results/recorded.json`. 1,375 late Rome fallback rows train "recorded vs copy"; 38 target rows (27 copy,
11 +24 h). AUC 0.70 for +24 h against copy, but the probabilities barely separate (mean 0.32 on +24 h rows, 0.22 on
copies). Arm A: +3.4e8 pooled, **-7.1e8 without the best 3, 3 of 10 months**; arm B -9.3e8. Fail. Whether a Rome
handler records a pushback time is not predictable enough from airline, stand, destination, runway and time.
