# Pre-registration: a Rome-specific label-mechanism classifier for the late fallback rows

Written 30 September 2026, **before any fit**. Owner's request of the same night (option 1: learn the coin-flip
pattern from 2025 data). NM-free, challenge data only. No candidate, no upload without the owner.

## Cell

Rome (LIRF) departures without AOBT_3 (the fallback lane) with delay = MVT - SCHED above 1 h, **excluding** the v69
rule rows (delay above 8 h and no local-midnight crossing, `research_500_midnight/rule.py`). 2025: 1,363 rows, 663
copies, 688 normal, 12 +24 h, every month present (47 to 320). 2026: 338 rows.

## Model, fixed now

LightGBM multiclass (copy / normal / +24 h, as `rule.label_kind`), 200 trees, learning rate 0.03, 7 leaves,
min_child_samples 20, lambda_l2 10, feature_fraction 0.9, bagging 0.8, seeds 177, 277, 377 averaged. Inputs, all present
in the ranking data: log delay; crossing (days); local (Europe/Rome) hour of SCHED and of MVT; weekday of SCHED; airline
(first three characters of FLIGHT_mvt), destination country (first two characters of ADES_mvt) and stand area (first
character of STAND_mvt) as LightGBM categoricals; runway. No month input (it cannot generalise to a held-out month).

Expected value per row: p_copy x delay + p_normal x t + p_+24 x (86,400 + t), with t the median normal-row label of the
training months.

## Arms

- A (registered decision arm): the average of the shipped value and the classifier's expected value.
- B (reported, not judged): the classifier's expected value alone.

## Test: leave one month out over 2025

For each month m, the classifier is fitted on the other eleven months' cell rows and applied to month m's. Reference:
the shipped fallback's six-pair out-of-sample value (`prod_anchor`). Gain = reference squared error minus arm squared
error. All must hold for arm A:

1. Pooled gain above zero.
2. Above zero after removing the 20 most favourable rows.
3. At least 8 of the 12 months above zero.
4. January and July both above zero.
5. Planning: pooled gain per row times 338, over 344,841, against v69's central 255.32, at least 0.1 s.

Reported: log loss and AUC for copy against the rows' shipped commitment, gain by crossing and delay band.

A pass leads to v70 = v69 plus arm A on the 2026 cell rows, with the classifier fitted on all twelve months, built and
verified like v69; the upload stays with the owner.

## Result, 30 September 2026: fail

`rome_classifier.py`, `results/rome_classifier.json`. The classifier ranks copies better than the shipped value's own
commitment (AUC 0.898 against 0.850, log loss 0.415). Arm A: pooled +1.39e8 (+0.20 s planned), but -1.45e9 without the
best 20, 7 of 12 months, January and July both negative. **Fail.** Arm B alone: -5.36e9. The loss sits in the 8 to 40 h
band (arm A -2.40e9), where the +24 h class has 12 training rows; the 1 to 8 h bands gain +2.54e9 together. That split
was seen after the run and is a post-hoc hypothesis, not a result.

## Addendum A, 30 September 2026: the 1 to 8 h slice as a candidate, post-hoc and disclosed

Written after the result above and **before any 2026 value is computed**. The slice was chosen after seeing the band
split, so it is not a confirmed result. Reasons it is still put forward: the 8 h bound is v69's pre-registered
threshold, and the +24 h class the classifier cannot learn (12 training rows) sits almost entirely above it (1 of 12
below 8 h). Evidence in the slice (same leave-one-month-out run, no refit): 1,325 rows, pooled +2.54e9, +9.6e8 without
the best 20, -1.2e8 without the best 50, **11 of 12 months positive, January +7.3e7 and July +5.0e8**. Day-block
bootstrap (315 days, 10,000 samples of whole days adding up to 2026's 326 rows): planning 1.47 / 3.51 / 5.84 s at the
5th / 50th / 95th percentile, 0.1 % of samples negative.

**Candidate v70** = v69 plus arm A on the 2026 slice: Rome fallback departures, delay above 1 h and at most 8 h, not a
v69 rule row. The classifier is fitted once on all 1,363 cell rows of 2025 (all bands, as in the test), t the median
normal label of those rows; new value (v69 value + expected value) / 2. Nothing else changes.

**Prediction fixed before any upload**, discounted by half for the post-hoc selection: central 255.32 - 1.75 = 253.57;
best 255.32 - 5.84 = 249.48; worst 255.32 + 1.0 = 256.32 (a 2026 draw below the bootstrap's range, allowing for the
selection and for 2026 differing from 2025). Upload only with the owner, once, and never adjusted on its result.

## Candidate v70, built and verified (30 September 2026)

`build_v70.py`, `verify_v70.py` (slice recomputed with zoneinfo; classifier refitted from scratch in a fresh process
reproduces every value exactly): v69 plus arm A on the 326 slice rows, all 326 change (mean +162 s, RMS 1,220 s, max
6,753 s); mean p(copy) 0.44 against a 2025 cell copy share of 0.49; everything else byte-identical to v69; NM-free;
SHA a0685ae5... Prediction as fixed in addendum A: central 253.57, best 249.48, worst 256.32.
