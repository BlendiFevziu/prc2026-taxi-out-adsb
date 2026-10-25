# Pre-registration: recalibrated expected values for Rome's rows without AOBT_3

Written 2 October 2026, **before any arm below is scored**. NM-free, 2025 labels only, no leaderboard information. No
upload without the owner.

**Disclosed:** the arms follow from `research_700_probe/calibration.py`, which already read the 2025 labels through a
leave-one-month-out run of the shipped Rome classifier. It showed the classifier overcautious overall (realised error
0.58 of believed) and wrong in the sparse tails: crossing rows of 8 to 14 h were 19 of 19 copies (classifier 0.84),
crossing rows above 14 h 11 of 19 "+24 h" (classifier 0.18). The test below is still leave-one-month-out, but the arm
idea was found on the same rows, so the prediction is discounted.

## Rows

Rome (LIRF) departures without AOBT_3, delay (MVT - SCHED) above 1 h. 2025: every month. 2026: every such row.
Kinds as `research_500_midnight/rule.py` `label_kind` (copy, normal, shift24); values per kind: copy = delay,
normal = t, shift24 = 86,400 + t, with t the median label of the training months' normal rows.

## Reference: the shipped logic, reproduced on 2025

prod_anchor (six-pair out-of-sample) for every row; v69 rule rows (delay above 8 h, crossing 0) = delay; v70 cell rows
(delay above 1 h and at most 8 h, not a rule row) = (prod_anchor + classifier expected value) / 2, classifier fitted on
the other months' cell rows (as `rome_classifier.py` arm A).

## Arms, fixed now

Each from probabilities fitted on the other eleven months only:

- **A** expected value from the classifier (`rome_classifier.fit_predict`, unchanged) fitted on all rows of the
  population above (delay above 1 h, crossing or not).
- **B** expected value from the empirical kind shares of the row's cell, cells = delay band (1-2, 2-4, 4-8, 8-14,
  above 14 h) x crossing (0, at least 1); an empty cell takes its band's shares.
- **C** (A + B) / 2.

The arm with the largest pooled gain over the reference is selected (selection among three is disclosed, not
corrected).

## Rules for the selected arm, all must hold

1. pooled gain over the reference above zero;
2. day-block bootstrap (5,000 draws, scheduled local day as block) 95 % lower bound above zero
   (`research_640_tail_rule/POLICY_tail_rule.md`);
3. at least eight of twelve months above zero, January and July each above zero;
4. planning: gain per row x 2026 rows, over 344,841, against v80's official 245.046, at least 0.1 s.

A pass leads to v83 = v80 with the selected arm (refitted on all twelve months) on the 2026 rows. Prediction fixed in
the build: central 0.5, best 1.0, worst 0.0 of the planned gain (discounted for the disclosed arm search).

## Result, 2 October 2026: fail on every rule

`rome_calibration.py`, `results/rome_calibration.json`. 1,375 rows (675 copy, 688 normal, 12 shift24). Reference (shipped
logic reproduced) 3,915 s RMSE; arm A 4,506, B 4,528, C 4,230. Selected C: -2.57e6 per row, day bootstrap
[-4.08e6, -0.64e6], 2 of 12 months positive, planning -5.28 s. Pure expected values lose to the anchor model, which
carries information the kind classifier lacks; the classifier's tail miscalibration is real but small next to that.
Nothing built.
