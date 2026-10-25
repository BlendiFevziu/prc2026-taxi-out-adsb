# Pre-registration: the build's corrector (pilot 6 design) plus five extra training months (pilot 11)

Written 19 October 2026, after pilot 9 (both parts passed; five_ens 202.166 stayed above pilot 6's 200.992, so its rebuild condition failed) and **before any fit for this test**. Changes go into dated addenda.

## 0. Why

The current 2026 file uses pilot 6's design: the corrector learns over the ensemble's own held-out predictions on January and July. Pilot 9 showed that five months beat two (+1.48 s and +1.73 s), but
its corrector learned over stage-1 LightGBM predictions only, and that lost about 1.2 s against pilot 6. This test keeps pilot 6 exactly and only adds rows.

## 1. Arms (January and July 2025, 339,046 main-lane rows, p1 = the ensemble's held-out predictions)

- **pilot6**: pilot 6's stack arm, its cached out-of-fold predictions (`stack_ens_stack_s{177,277}.npy`), 200.992.
- **plus5**: the same four day folds (`crc32(date) % 4`), the same corrector settings and features, seeds 177 and 277; for fold k the training rows are the January/July rows outside fold k (ensemble p1) **plus all
  February, June, March, May and December rows** (stage-1 LightGBM p1 of pilots 7 and 9, the seed's own column). No new feature marks the source of p1.

## 2. Decision (plus5 against pilot6, all must hold)

1. plus5 beats pilot6 in each seed; 2. positive in January and in July; 3. positive after removing the 100 most favourable rows; 4. airport-day block bootstrap 95 % interval above zero; 5. RMSE gain at least 0.3 s;
6. plus5 beats the ensemble base.

## 3. Meaning

Pass: the 2026 file is rebuilt with the corrector trained on January and July (ensemble p1) plus every further 2025 month with tracks (stage-1 LightGBM p1, leave-out for that month), and the build report records
it. Fail: the current file stays, and more months are not used in the build.

## Addendum A, 19 October 2026: pilot 11 fails on rules 2 and 3

`pilot11_screen.py`, `results/pilot11_screen.json`, log `.cache/research_730/pilot11.log`. Extra rows 829,647 (February, June, March, May, December). pilot6 200.992, **plus5 200.483 (+0.51 s)**; seeds +197 / +212
MSE per row, bootstrap [+32, +382], but **January -147** (July +491) and **-62 without the best 100 rows**. By airport against pilot 6: EDDM +791, LSZH +790, EDDF +477, LEBL +450, EHAM +329, LIRF +141, LFPG +27,
LTFM -113, LEMD -153, EGLL -168. **Verdict: fail.** The current 2026 file stays, and more months are not used in the build in this form. Reading: rows whose p1 comes from a different, weaker model dilute
the corrector the ensemble's own rows taught; using further months properly needs the ensemble's held-out predictions for those months.
