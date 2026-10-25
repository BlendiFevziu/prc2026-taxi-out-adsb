# Pre-registration: daily reconciliation without Rome, tested on February and June 2025 (pilot 8b)

Written 15 October 2026, after pilot 8 failed and **before any February or June 2025 prediction exists**. Changes go into dated addenda.

## 0. Why

Pilot 8 (January and July 2025) failed, entirely because of LIRF; every other airport gained. Leaving LIRF unchanged was computed afterwards (-0.62 s) and is therefore not evidence. This registration fixes
that variant and tests it on months pilot 8 never read.

## 1. Method

Identical to `PREREG_daily_reconcile.md` section 1, with one change: **LIRF rows are never shifted** (they still enter their own day's sum, which is then simply not used because no LIRF row moves).
Months: **February and June 2025**. Predictions `p`: the corrector of pilot 7 applied to February and June (direction 1_7 -> 2_6, mean of its two seeds; corrector trained on January and July only), so
neither the corrector nor `lambda` ever saw these months' labels; fallback rows use the same stand-in as pilot 8. `lambda` is chosen on the other month of the pair (February from June and June from February).

## 2. Arms and decision

Arms as pilot 8 (reconciled, three day-permuted controls, no step). All six rules of pilot 8 must hold on the February and June main-lane rows, rule 6 reading "`lambda` above zero in both months".

## 3. Meaning

Pass: the reconciliation without LIRF is the last step of the rebuild. Fail: it is dropped.

## Addendum A, 16 October 2026: pilot 8b fails on rule 4 only

`pilot8b_screen.py`, `results/pilot8b_screen.json`. February and June 2025 main-lane rows (323,262), predictions of pilot 7's January/July-trained corrector (mean of the two seeds' predictions: 199.668).
Reconciled **198.312 (+1.36 s)**; the controls chose `lambda = 0` in both months, the real figures 0.75 (February) and 0.25 (June). February +1,010 MSE per row, June +168; +104 without the best 100 rows.
Rules 1, 2, 3, 5 and 6 hold; **rule 4 fails: airport-day bootstrap [-110, +1,282] includes zero.** By airport: LTFM +3,332 carries most of it, LFPG +426, LEMD +237, EGLL +184, LEBL +180, EDDM +138,
EDDF +116, LSZH +67, LIRF 0 (not moved), EHAM -197. **Verdict: fail.** The step is not adopted. Reading: the effect is positive and sizeable but rests on few airport-days (LTFM, February); a confirmation needs
further months with tracks, and the rule and thresholds stay as registered for that.
