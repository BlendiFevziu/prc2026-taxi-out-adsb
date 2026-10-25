# Pre-registration: daily reconciliation without Rome, confirmation on March, May and December 2025 (pilot 10)

Written 19 October 2026, **before any March, May or December 2025 corrector prediction exists** (pilot 9 is running, but its part A and B predict only January, February, June and July).
Changes go into dated addenda.

## 0. Why

Pilot 8b (February and June) passed five of six rules and failed rule 4 (airport-day interval [-110, +1,282]); its addendum said a confirmation needs further months with tracks, with rule and thresholds
unchanged. We now hold March, May and December 2025.

## 1. Method

Identical to `PREREG_daily_reconcile_v2.md` (pilot 8b, code `pilot8b_screen.py`): LIRF rows never shift, shifted rows are main-lane rows with `p <= 3,600 s`, bisection for `s*`, `s = lambda * s*` capped at
+-300 s, prediction `max(p + s, 300)`, fallback stand-in the airport's median main-lane prediction of that month, three day-permuted controls (seeds 16001 to 16003).

Two changes, both forced by having three months instead of two:
- predictions `p`: the pilot-5 corrector trained on **January, February, June and July** (stage-1 LightGBM p1 of pilot 7), applied to March, May and December (p1 from pilot 9's leave-group-out stage 1), mean of
  seeds 177 and 277. Neither the corrector nor stage 1 saw these months' labels.
- `lambda` for each month is chosen on the **other two months together**, by main-lane RMSE.

Rows: the main-lane rows of pilot 9's table (`table_3_5_12_stand.parquet`, 2025-12-31 dropped).

## 2. Decision

The six rules of pilot 8b on these rows, rule 2 and rule 6 reading "each of the three months". Pass: the reconciliation without LIRF becomes the last step of the 2026 file (applied with `lambda` chosen on all five
months February, June, March, May, December pooled). Fail: it is dropped for good.

## Addendum A, 19 October 2026: pilot 10 fails on rule 5 only

`pilot10_screen.py`, `results/pilot10_screen.json`. 506,385 main-lane rows of March, May and December 2025; corrector trained on January, February, June and July: **172.952** (stage 1 193.634). Reconciled **172.681
(+0.27 s)**; lambda 0.25 / 0.25 / 0.5, every control chose 0. March +112, May +117, December +47 MSE per row, +63 without the best 100, bootstrap [+5.5, +155.9]. Rules 1, 2, 3, 4 and 6 hold; **rule 5 fails
(0.27 s < 0.3 s)**. By airport: LTFM +346, EGLL +257, LFPG +165, LEBL +152, LEMD +147, EDDF +82, LSZH +30, LIRF 0, EHAM -67, EDDM -401. **Verdict: fail. Dropped for good, as section 2 says.** The effect is real
(the controls never pick a step, and this time the interval clears zero) but smaller than the registered minimum on months the corrector already predicts well.
