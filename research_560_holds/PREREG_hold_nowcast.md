# Pre-registration: a ground-hold nowcast for rows without AOBT_3 (outside Rome)

Written 30 September 2026, **before any correction is estimated or scored**. Owner's request ("do 1": the rows
without AOBT_3). NM-free, no fit of the fallback model. No upload without the owner.

## Why

Genuine holds (label above 1.5 h, neither a copy nor +24 h) carry 7.4 % of fallback squared error in 2025; the shipped
fallback predicts a median 0.85 h on them against 1.87 h true. A signal present at rank time, the share of main-lane
departures at the same airport within +/-2 h of the row's take-off whose MVT - AOBT_3 exceeds 2,700 s ("share2"),
raised the hold rate about tenfold in its top decile (descriptive, `experiments.csv` 30 September, AUC 0.67). The
shipped fallback's neighbour anchor uses the median of the same neighbours, not their tail.

## Rule, fixed now

For fallback rows outside LIRF (Rome's residuals are copy-driven and handled by v69/v70): bands share2 in [0.1, 0.2)
and [0.2, 1]; below 0.1 or without neighbours nothing changes. In each band the correction is the mean residual
(label minus the shipped value) of the estimation rows, shrunk by n / (n + 50). New value max(shipped + correction, 300).

## Test: six pairs, leave-pair-out

For each pair a_b (validation months a and b), the band corrections are estimated on the other five pairs'
out-of-sample rows (`prod_anchor`) and applied to the pair's own rows. Rules, all must hold: pooled gain over all
non-Rome fallback rows above zero; above zero without the best 20 rows; at least four of six pairs above zero;
planning: gain per non-Rome fallback row times 4,400 (2026 non-Rome fallback rows not already overridden by the v74/v75
detectors, rounded), over 344,841, against v75's official 249.3908, at least 0.1 s.

A pass leads to v76 = v75 plus the correction on those 2026 rows, with the bands estimated on all six pairs.

## Result, 30 September 2026: fail

`hold_nowcast.py`, `results/hold_nowcast.json`. 20,982 non-Rome fallback rows, 286 in [0.1, 0.2) and 138 in [0.2, 1];
hold rate 8.4 % and 13.8 % against 0.46 % elsewhere, so the signal is real, but it touches 2 % of the rows: pooled
+947 per row, **-1,429 without the best 20, 3 of 6 pairs, planning +0.02 s**. Fail; nothing built.
