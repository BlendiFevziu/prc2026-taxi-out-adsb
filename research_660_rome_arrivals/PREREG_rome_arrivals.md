# Pre-registration: arrival recording signals for the Rome copy classifier (v70 slice)

Written 1 October 2026, **before any fit with these inputs**. Owner: "lets go all out on the extreme fallback rows".
NM-free, challenge data only. No upload without the owner.

## What was seen first (disclosed)

A 2025 replica of the shipped Rome logic (prod_anchor, v69 rule, v70 classifier left out by month) puts 39 % of the
remaining Rome fallback squared error in the v70 slice's copy/normal coin flips (1,324 rows). Descriptive crosstabs on
2025 labels, seen before this registration: Rome arrivals carry a copy mechanism of their own (BLOCK = SCHED within
60 s on 11.8 % of arrivals, visible in the ranking data because arrival BLOCK times are not hidden). Late Rome fallback
departures are copies on 68 to 89 % of rows when more than a quarter of the same airline's arrivals that local day are
copies, against 37 to 51 % otherwise; the day-level share orders them from 42 % to 62 % by quintile; the inbound
arrival at the same stand is weak (58 % against 45 %). These were seen on the same 2025 rows the test below uses, so
the test is a check of transfer across months, not an untouched confirmation.

## Inputs added, fixed now (all from arrival rows, no departure label)

From Rome (ADES LIRF) arrivals, copy = |BLOCK - SCHED| <= 60 s, local date of SCHED (Europe/Rome):
1. `arr_copy_al_day`: copy share among the same airline's (first three characters of FLIGHT_mvt) arrivals that local
   day (missing when none);
2. `n_arr_al_day`: their number;
3. `arr_copy_day`: copy share among all Rome arrivals that local day;
4. `in_copy`: whether the last arrival at the same stand, landing (MVT) within 18 h before the departure's SCHED, was a
   copy (missing when none).

## Test

Same cell, same leave-one-month-out loop, same LightGBM settings and expected-value formula as
`research_510_rome_classifier/rome_classifier.py`, applied as v70 is: on the slice (delay above 1 h and at most 8 h, not
a v69 row), value (prod_anchor + expected value) / 2. Reference: the same with the original nine inputs (today's v70).
Gain = reference squared error minus candidate squared error. All must hold:

1. Pooled gain above zero.
2. Day-block bootstrap (local SCHED date, 5,000 draws, seed 0) 95 % lower bound above zero.
3. At least 8 of 12 months above zero.
4. January and July both above zero.
5. Planning: gain per row times the 2026 slice rows (326), over 344,841, against v81's central 245.020, at least 0.1 s.

Reported: AUC for copy with and without the new inputs; the 2026 arrival copy share (no label needed).

A pass leads to v82 = v81 with the slice rows recomputed by the classifier with the new inputs, fitted on all twelve
months (the v70 build recipe), built and verified like v70.

## Result, 1 October 2026: fail (January and July)

`rome_arrivals.py`, `results/rome_arrivals.json`. Inputs present on 93 to 100 % of the 1,363 cell rows; 2026 arrival copy
share 11.7 % against 12.1 % in 2025. AUC for copy 0.898 -> 0.906 (slice 0.899 -> 0.907). On the 1,325 slice rows: pooled
+214,119 per row, day bootstrap 95 % [+15,600, +433,595] over 315 days, 8 of 12 months positive, planning +0.41 s, but
**January -6.2e6 and July -1.97e7**, the two months 2026 is scored on. Fail as registered; nothing built. The arrival
signal is real but small, and the gain sits in spring and summer months the score does not contain.
