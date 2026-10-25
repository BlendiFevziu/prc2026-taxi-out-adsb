# Pre-registration: a month-matched corrector for February and June (pilot 14)

Written 24 October 2026, after pilot 13 and **before any fit for this test**. Changes go into dated addenda.

## 0. Why

Pilots 11 to 13 agree that the corrector does best when it learns from its own months. The current 2026 file applies one corrector, trained on January/July 2025, to all four 2026 months. For February and
June 2026 a corrector trained on February/June 2025 is now possible, because the ensemble's own held-out predictions exist for those months (`.cache/ensemble_holdout_2_6`, pilot 12).

## 1. Arms (February/June 2025 main-lane rows, p1 = the ensemble's held-out prediction; pilot-5 corrector settings and features; seeds 177 and 277)

- **cross**: trained on all January/July 2025 rows (p1 = `ensemble_baseline`), applied to February/June. This is what the current file does to February/June 2026.
- **matched**: trained on February/June itself with **contiguous week blocks** as folds (days 1-7, 8-14, 15-21, 22-end of each month), so that a test day's neighbours in time are not in its training set.
  Day folds by `crc32(date)` (pilot 12's cached `pair_26`) are reported, not judged.
- Reported, not judged, the mirror case: January/July with matched week-block folds against a corrector trained on February/June.

## 2. Decision (matched against cross on February/June, all must hold)

1. matched beats cross in each seed; 2. positive in February and in June; 3. positive without the 100 most favourable rows; 4. airport-day block bootstrap 95 % interval above zero; 5. RMSE gain at least 0.3 s;
6. matched beats the ensemble base.

Known limit, fixed now: a week block is much closer in time than a year, so a pass shows that matching the months helps within a year, and is necessary rather than sufficient for 2026.

## 3. Meaning

Pass: a second 2026 file is built (`final_rebuild_2026_matched.parquet`), identical to the current one except that the February/June 2026 main-lane rows are corrected by the February/June 2025 corrector
(all rows, seeds 177 and 277 averaged, over the 2026 base); it becomes the preferred file, with this limit recorded. Fail: the current file stays.

## Addendum A, 24 October 2026: fails on rules 3 and 4; the gain sits in a few extreme rows

`pilot14_screen.py test`, `results/pilot14_screen.json`, log `.cache/research_730/pilot14.log`. **February/June (judged)**, 323,262 rows, ensemble 209.612: cross (January/July corrector) 193.585,
**matched (week blocks) 191.621, +1.96 s**; seeds +710 / +803 MSE per row, February +1,348, June +290, but **-30 without the 100 most favourable rows** and airport-day bootstrap **[-86, +1,777]**. By airport:
LTFM +4,185 carries it, LEMD +1,130, EGLL +569, LIRF +475, LEBL +221, LSZH +96, EHAM +94, LFPG -78, EDDM -125, EDDF -127. Rules 1, 2, 5, 6 hold; **rules 3 and 4 fail. Verdict: fail.** The current 2026 file stays;
`build` was not run. Reported: crc32 day folds 191.156; mirror (January/July, matched against the February/June corrector) 203.898 -> 201.134 (+2.76 s), all six rules hold there, but it was registered
as reported only and does not change the January/July rows, which the current file already corrects with a January/July corrector.
