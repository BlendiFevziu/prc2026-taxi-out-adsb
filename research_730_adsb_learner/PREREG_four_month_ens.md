# Pre-registration: the corrector over the ensemble's own predictions on four months instead of two (pilot 12)

Written 19 October 2026, **before the February/June ensemble predictions exist** (their training, `research_740_ensemble_holdout/`, starts now). Changes go into dated addenda.

## 0. Why

Pilot 9 showed more training months help (+1.48 s); pilot 11 showed they hurt January when their p1 comes from a weaker model. With the frozen v45 candidate-stage ensemble trained on the ten months other than
February and June, both pairs have p1 of the same kind.

## 1. Inputs

January/July: `ensemble_baseline.parquet` (216.773, as pilot 6). February/June: the main-lane rows of `.cache/ensemble_holdout_2_6/candidate/holdout_predictions.parquet`, joined to `table_2_6_stand.parquet`
by `MVT_ID_mvt`; every table row must find a prediction. Its RMSE is reported, not judged.

## 2. Arms (all 662,308 main-lane rows of the four months, day folds `crc32(date) % 4` as pilot 6, seeds 177 and 277, pilot-5 corrector settings and features)

- **pair**: for each pair separately, pilot 6's design (training on the other folds of the same pair). On January/July this reproduces pilot 6's cached predictions, which are reused.
- **four**: for fold k, training on the rows outside fold k of all four months.

## 3. Decision (four against pair, all must hold)

1. four beats pair in each seed; 2. positive in each of the four months; 3. positive after removing the 100 most favourable rows; 4. airport-day block bootstrap 95 % interval above zero; 5. RMSE gain at least 0.3 s;
6. four beats the ensemble base.

## 4. Meaning

Pass: the 2026 file is rebuilt with the corrector trained on all four months over the ensemble's own predictions; the same step is then repeated for March, May, December and the months still downloading.
Fail: the current file stays.

## Addendum A, 20 October 2026: pilot 12 fails on rule 2 only

February/June ensemble (`.cache/ensemble_holdout_2_6`, frozen v45 candidate stage trained on the ten other months, tune 1,026 -> 1,078 rounds, stage finished 19 October 23:14 UTC): main-lane RMSE **209.612**
(January/July 216.773). `pilot12_screen.py`, `results/pilot12_screen.json`, log `.cache/research_730/pilot12.log`. Four months, 662,308 rows: base 213.308, **pair 196.253, four 195.837 (+0.42 s)**; both seeds
+163 MSE per row, +14 without the best 100 rows, bootstrap [+47, +272]. By month: February +152, June +289, July +259, **January -93** (178.567 -> 178.827). By airport: LIRF +426, EDDM +487, LSZH +345, EHAM +307,
LEBL +177, EDDF +165, LFPG +128, EGLL +95, LEMD -43, LTFM -163. Rules 1, 3, 4, 5, 6 hold; **rule 2 fails. Verdict: fail.** The current 2026 file stays.
Reading, not evidence: pilots 11 and 12 both lose January and only January when other months join the training, whether their p1 is weaker (pilot 11, -147) or of the same kind (here, -93). The cause is
therefore not the p1 source. Any month-dependent remedy would be chosen after seeing this, so it needs its own registration and months not used here.
