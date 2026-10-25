# Pre-registration: neighbouring months against all months as extra corrector training, judged on fresh winter months (pilot 13)

Written 23 October 2026, **after starting the November/December ensemble run (`.cache/ensemble_holdout_11_12`, research_740) and before any of its predictions exist**. November tracks are still
downloading. Changes go into dated addenda.

## 0. Why

Pilots 11 and 12 lost January, and only January, when other months joined the corrector's training (-147 and -93 MSE per row), whatever the source of p1. Two readings: extra months from other seasons pull the
corrector away from a winter month, or January is special. The remedy "train on neighbouring months only" was thought of after seeing this, so it is judged only on months no earlier test used: November and
December 2025, January's winter neighbours.

## 1. Inputs

p1 for November and December: the main-lane rows of `.cache/ensemble_holdout_11_12/candidate/holdout_predictions.parquet` (frozen v45 candidate stage trained on the ten other months). p1 for January/July and
February/June: as pilot 12. ADS-B and stand columns: December from `table_3_5_12_stand.parquet` (2025-12-31 dropped, no track day), November computed with the same code as pilot 9 once all 30 November
track days exist (the run refuses otherwise). Every table row must find a prediction.

## 2. Arms (November and December main-lane rows; day folds `crc32(date) % 4`; pilot-5 corrector settings and features; seeds 177 and 277)

- **own**: for fold k, training on the November/December rows outside fold k (pilot 6's design on this pair).
- **neighbour**: own plus all January rows (January is the only adjacent month with ensemble p1; October has none).
- **all**: own plus all January, February, June and July rows.

## 3. Decisions (six rules each: new beats old in each seed; positive in November and in December; positive without the 100 most favourable rows; airport-day bootstrap 95 % interval above zero; RMSE gain at least 0.3 s; new beats the ensemble base)

- **B, neighbour against own: the judged test.** Pass: neighbouring months help a winter month; the next step is a 2026 build with one corrector per month trained on that month's pair plus its adjacent months with
  ensemble p1, registered separately. Fail: neighbouring months do not help.
- **A, all against own: reported, and read against pilots 11/12.** If it fails rule 2 or loses here too, the January loss is a seasonal effect, not a January peculiarity.

If B fails and A fails, the current 2026 file stays and extra months are not used in the corrector.

## Addendum A, 24 October 2026: both arms fail; extra months hurt the winter months

November/December ensemble (`.cache/ensemble_holdout_11_12`, finished 24 October 00:57 UTC): main-lane RMSE 195.788 over the stage's rows; on the 320,397 table rows (November 160,976, matched 84.3 %;
December 159,421) **194.382**. `pilot13_screen.py`, `results/pilot13_screen.json`, log `.cache/research_730/pilot13.log`.
**own 164.655** (+29.7 s over the ensemble). **B, neighbour (own + January): 164.888, -0.23 s**; seeds -96 / -57 MSE per row, November -146, December -8, -216 without the best 100, bootstrap [-188, +20].
Only rule 6 holds. **Verdict B: fail.** **A, all (own + January, February, June, July): 164.914, -0.26 s**; November -195, December +25, bootstrap [-280, +82]; fails as well.
Reading: on fresh months, adding other months to the corrector's training makes it worse, with or without seasonal proximity; together with pilots 11 and 12 (January -147 and -93), the corrector works best trained on
its own month pair only. Extra months are not used in the corrector; the current 2026 file stays.
