# Pre-registration: reconciling each airport-day with EUROCONTROL's published pre-departure delay (pilot 8)

Written 15 October 2026, **before any reconciliation is computed on predictions**. Same conditions as the earlier pilots. Changes go into dated addenda.

## 0. Why, and what is already known

EUROCONTROL publishes per airport and day the all-causes pre-departure delay (`All_Pre-Departure_Delay.xlsx`, `DLY_ALL_PRE_2`, minutes; organiser-cleared as open data, `research_390_aiu/NOTE.md`).
Checked on 15 October against all 2025 labels (3,650 airport-days): it equals the day's sum of `max(0, off-block - schedule)` with correlation **0.99989** (keyed by the off-block UTC date)
and **0.99964** (keyed by the take-off UTC date, which is known at prediction time), median mismatch 0.2 s per flight, departure counts equal. Since off-block = take-off - taxi-out, the figure is a near-exact
day-level constraint on the predictions. The earlier study (`research_390_aiu/PREREG_daily.md`) used the same files only as model columns (pass on 1_7, fail on 3_9); this tests the constraint.
Three leading teams report it: vigorous-whistle -1.33 s officially, strong-crane -0.8 s, versatile-watermelon 0.72 s on twelve months.

## 1. Method (fixed now)

Rows of an airport and take-off UTC date: all departures of the training files (main lane and fallback). Prediction `p` per row: for main-lane rows the pilot-6 stack (mean of its two seeds, out of fold);
for fallback rows a label-free stand-in, the airport's median main-lane stack prediction of that month. Gap `G = take-off - schedule`. Shifted rows: main-lane rows with `p <= 3,600 s`; other rows keep `p`.
Find the common shift `s*` with `sum max(0, G - p - s*) = 60 * DLY_ALL_PRE_2` by bisection on [-1,800, +1,800] s (clamped at the ends). Apply `s = lambda * s*`, capped at +-300 s, to the shifted rows, prediction
`max(p + s, 300)`. `lambda` from the grid {0, 0.25, 0.5, 0.75, 1}, chosen on the **other** month (January from July and July from January) by main-lane RMSE.

## 2. Arms

**reconciled** (the published figure), **control_k, k = 1..3** (the published figures permuted across the days of the same airport and month, the same procedure, `lambda` chosen the same way), **stack** (pilot 6, no step).

## 3. Decision (all must hold; judged on the 339,046 main-lane rows)

1. reconciled beats the mean control; 2. reconciled beats stack in both months; 3. positive against stack after removing the 100 most favourable rows; 4. airport-day block bootstrap 95 % interval of the gain over
stack above zero; 5. RMSE gain over stack at least **0.3 s**; 6. the chosen `lambda` is above zero in both months. Reported, not judged: gain by airport, the distribution of `s`.

## 4. Meaning

Pass: the reconciliation is the last step of the rebuild. Fail: the constraint does not help on top of the corrector at this form; recorded, not tuned.

## Addendum A, 15 October 2026: pilot 8 fails

`pilot8_screen.py`, `results/pilot8_screen.json`. Main-lane RMSE: reconciled 201.354 against stack 200.889 (**-0.47 s**); the controls chose `lambda = 0` in every month (as they should), the real figures
0.25 (January) and 0.5 (July). January +28 MSE per row, July -363; airport-day interval [-679, +150]. Rules 1 to 5 fail. **Verdict: fail.** By airport (MSE per row against stack): LIRF **-5,665**, every
other airport positive (LTFM +762, LFPG +547, LEBL +443, EDDM +245, EDDF +98, EGLL +93, LSZH +74, LEMD +4, EHAM +3). Rome's copied and date-shifted records carry most of the day's published delay,
so a common shift of its ordinary rows absorbs a discrepancy that belongs to a few heavy rows. **Post-hoc diagnostic, not a result:** leaving LIRF unchanged would have given 200.267 (-0.62 s). This
variant is pre-registered separately (`PREREG_daily_reconcile_v2.md`) and tested on months this pilot never read.
