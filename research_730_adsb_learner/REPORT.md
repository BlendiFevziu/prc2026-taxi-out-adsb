# The ADS-B model after the competition: what was built, what was tested, what is settled

State of 24 October 2026. Owner's own project after the PRC Data Challenge 2026 closed on 14 October. Nothing here was uploaded, no leaderboard value was used, and every
model is free of third-party NM data. Each result below has a pre-registration in this folder (`PREREG_*.md`, with its outcome in a dated addendum), a script (`pilot*_screen.py`)
and a JSON result in `results/`. The running log is `../experiments.csv`.

## 1. The model

Two stages, taxi-out time per departure (`MVT_TIME - BLOCK_TIME`).

1. **Base.** The frozen four-month base (`.cache/research_800/base_final_predictions.parquet`): the v45 all-year ensemble (5 LightGBM, 3 CatBoost, 5 neural nets, 10 airport models,
   reporting gate) with the v47 sequence nets, the anchored fallback lane, the copy gate and the Rome rules.
2. **ADS-B corrector.** A residual LightGBM over the base's prediction for every main-lane departure (rows with an `AOBT_3` clock), using 12 columns from adsb.lol ground tracks matched
   by callsign (first and last ground point, speeds, stops, movement start, distance, gaps; `features_adsb.py`) and 6 stand-readout columns. Settings: learning rate 0.03, 63 leaves,
   min_child_samples 100, subsample and colsample 0.8, 600 rounds, two seeds averaged. Trained on all January/July 2025 rows over the ensemble's own held-out predictions.

The 2026 file: `.cache/research_730/rebuild_2026/final_rebuild_2026.parquet`, SHA-256 `acca1ea3c76e7adf2d80016aeaed9a9d18000f23944d4a45201a3c1029d093d5`, 670,790 rows,
661,713 main-lane rows corrected, fallback rows equal to the base, format checks pass (`BUILD_2026.md`). It has no score, because no 2026 label exists.

## 2. What the corrector is worth (2025, held-out)

| Months | Ensemble alone | With the corrector | Gain | Pilot |
|---|---:|---:|---:|---|
| January/July (day folds) | 216.77 | 200.99 | +15.8 s | 6 |
| February/June (own day folds) | 209.61 | 191.16 | +18.5 s | 12 |
| November/December (own day folds) | 194.38 | 164.66 | +29.7 s | 13 |
| Trained on one pair, applied to another (stage-1 LightGBM base) | 220.24 | 205.04 | +15.2 s | 7 |

For comparison, the best scored file during the competition was v80, 245.046 s on the leaderboard (all rows, fallback lane included). The tables above are main-lane RMSE on
2025 months, so they are not the same scale as a leaderboard score.

## 3. What was tested after the 2026 file, and failed (19 to 24 October)

Every test kept the six rules: better in each seed, better in each month, better without the 100 most favourable rows, airport-day bootstrap interval above zero, at least 0.3 s, better than the base.

| Pilot | Question | Result |
|---|---|---|
| 9 | Five training months instead of two (stage-1 LightGBM base) | Pass, +1.48 s; but 202.17 stays above pilot 6's 200.99 |
| 10 | Daily delay reconciliation on March, May, December | +0.27 s, under 0.3 s; dropped for good |
| 11 | Pilot 6 plus five extra months | +0.51 s, January worse (-147 MSE per row) |
| 12 | Four months over the ensemble's own predictions | +0.42 s, January worse (-93) |
| 13 | November/December plus January, or plus four months | -0.23 s and -0.26 s |
| 14 | February/June corrected by its own months instead of January/July | +1.96 s, but negative without the best 100 rows (Istanbul carries it) |
| 15, 15b | Corrector size, 11 configurations | 600 rounds x 63 leaves is best on both sides |
| (C) | ADS-B for the fallback lane | Matched rows hold 0.5 % of the lane's error; closed |

The settled conclusions:
- **The corrector learns best from its own months.** Adding other months, near or far, never passed (pilots 11 to 14).
- **Its size is right.** Larger trees over-fit by up to 4.8 s, smaller ones lose up to 2.9 s.
- **The fallback lane cannot be fixed with tracks.** Its error sits in a few giant records without callsigns.

Earlier, from the competition-era work: the plan-cliff rule from the versatile-watermelon repository gave no gain; the daily reconciliation failed on January/July (Rome) and on February/June
(rule 4); learned arrival stands were deferred.

## 4. Data held

- adsb.lol tracks (ODbL, attribution "adsb.lol contributors"), `../../adsb-lol/tracks/`: **2025: 363 of 365 days.** Missing 2025-10-14 (its published archive fails to decompress on two independent
  downloads) and 2025-12-31 (no release). **2026: 119 days** of the four final-phase months (11 June 2026 has no archive).
- Ensemble held-out predictions (frozen v45 candidate stage, `research_740_ensemble_holdout/make_run.py`): January/July (September run), February/June and November/December
  (`.cache/ensemble_holdout_2_6`, `.cache/ensemble_holdout_11_12`).

## 5. What would still move the number

Within our own data, the reachable levers are used up. The ones left need something new:
- **2026 labels**, for a real score of the 2026 file (the organisers no longer publish one).
- **More callsigns for the fallback lane**, from a source other than the challenge's flight table.
- **A better base**, since the corrector adds a fairly steady 15 to 30 s on top of whatever it corrects. That means retraining the ensemble itself, which is the expensive end of the project.

## 6. Reproduce

From the repository root: `python3 -u -B research_730_adsb_learner/build_2026_final.py table`, then `... build`. Each pilot: `python3 -u -B research_730_adsb_learner/pilotN_screen.py <stage>`,
with the stages named in its docstring. Expensive fits take the shared lock `research_300/heavy_lock.py`; the ensemble runs follow `research_740_ensemble_holdout/make_run.py`.
