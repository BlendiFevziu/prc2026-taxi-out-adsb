# Pre-registration: does the ADS-B corrector transfer across months? (pilot 7)

Written 15 October 2026, **before any fit for it and before the February and June 2025 tracks are complete**. Same conditions as the earlier pilots (private research, no leaderboard, no upload).
Changes go into dated addenda.

## 0. Why

Pilots 5 and 6 trained and tested the corrector on different days of the same two months. In use, the corrector is trained on the months that have tracks and applied to other months
(2026). This pilot trains it on one month pair and tests it on another: January and July 2025 against February and June 2025, which are the 2025 analogues of all four final-phase months.

## 1. Stage 1 (leave-pair-out, no ADS-B)

For each pair P in {(1, 7), (2, 6)}: one LightGBM on the 91 release features, trained on the main-lane rows (`hat_aobt == 1`, label present) of the **ten other** 2025 months, predicting the rows
of P. Parameters as pilot 1 (learning rate 0.03, 255 leaves, min_child_samples 50, subsample and colsample 0.8, 1,200 rounds, target `y - ersatz_s`, prediction `max(ersatz_s + model, 300)`),
seeds 177 and 277. Release table: `.cache/clock_queue_release_20260912/data/months_1_2_3_4_5_6_7_8_9_10_11_12.parquet`.

## 2. Stage 2 (the corrector, cross-month)

The pilot-5 corrector (`pilot5_screen.stage2` settings: residual LightGBM, learning rate 0.03, 63 leaves, min_child_samples 100, subsample and colsample 0.8, 600 rounds; features `p1`, `ersatz_s`,
`stunde`, `ADEP_mvt` + the 12 ADS-B and 6 OSM stand columns), **trained on all rows of one pair and applied to the other pair**, in both directions. ADS-B columns for February and June are computed with
the same code once all 58 track days of February and June 2025 are extracted; the run waits for that.

Arms per direction and seed: **stack** (with the 18 columns), **control_k** (the 18 columns permuted jointly within airport and day in **both** the training and the applied pair, k = 1..3), **base** (`p1`).

## 3. Decision (all must hold; judged on the four held-out months together, each predicted by a corrector that never saw its pair)

1. stack beats the mean control in each seed; 2. positive in **each** of the four months; 3. positive after removing the 100 most favourable rows; 4. airport-day block bootstrap 95 % interval above zero;
5. RMSE gain over the mean control at least **0.3 s**; 6. stack beats base. Reported, not judged: the gain per direction, per month and per airport, and its size against pilot 5's within-month +17.6 s.

## 4. Meaning

Pass: the corrector can be trained on the 2025 months with tracks and applied to 2026; the next step is the 2026 build (ensemble + corrector) with the corrector trained on all four months. Fail: the gain does
not survive a change of months, and the within-month results overstated it.

## Addendum A, 16 October 2026: pilot 7 passes all six rules

`pilot7_screen.py`, `results/pilot7_screen.json`, log `.cache/research_730/pilot7.log`. February/June 2025 table: 323,262 main-lane rows (all 58 track days), 77.8 % matched, stand readout 9.8 %.
Stage 1 (trained on the ten other months): January/July 223.58, February/June 216.68. Four held-out months together (662,308 rows), mean of two seeds: **stack 205.041**, mean control 221.702,
base 220.241; gain **+15.20 s** over the base, +16.66 s over the controls. By direction: corrector trained on January/July applied to February/June 216.68 -> **200.20**; trained on February/June applied to
January/July 223.58 -> **209.55**. Both seeds positive (+7,079 / +7,142 MSE per row), every month positive (January +7,018, February +6,510, June +7,608, July +7,162), +6,760 without the best 100 rows,
airport-day bootstrap [+6,645, +7,581]. By airport against the base (MSE per row): EHAM +13.2k, LSZH +12.9k, EDDM +12.1k, LEBL +10.8k, EDDF +10.6k, EGLL +4.2k, LEMD +3.8k, LIRF +3.0k, LFPG -0.3k, LTFM -0.5k.
**Verdict: pass.** The cross-month gain (+15.2 s) equals the within-month gain (+15.8 s in pilot 6 on a stronger base, +17.6 s in pilot 5 on a weaker one), so the day-fold pilots did not overstate it. The corrector
can be trained on the 2025 months with tracks and applied to 2026.
