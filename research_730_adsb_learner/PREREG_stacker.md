# Pre-registration: the ADS-B columns in a second-stage corrector (pilot 5)

Written 15 October 2026, **before any code for it is run and before any fit**. Same conditions as the earlier pilots (private research, no leaderboard, no upload).
Changes go into dated addenda.

## 0. Why

The one-stage tests (pilots 1, 2, 4) put the ADS-B columns into the base model, which then has to be trained on rows that carry them. In a full rebuild the base models are trained
on twelve months, and we hold tracks for only some of them. The leading teams (jovial-uniform, amazing-marshmallow) use a second stage: a model over the out-of-fold predictions of the
first stage plus the ADS-B columns, "the one large step". This pilot tests whether that form keeps the gain, because then it can sit on top of any first stage, including the shipped one.

## 1. Design

Rows, folds, seeds and the cached out-of-fold predictions `p1` of the plain 91-feature base arm are those of pilot 1 (339,046 main-lane January and July 2025 departures, four day-blocked folds,
seeds 177 and 277). Stage 2 is one LightGBM per fold and seed on the residual `y - p1`, trained on the other folds' rows (their `p1` are out-of-fold too), predicting `max(p1 + r, 300)`.
Features: `p1`, `ersatz_s`, `stunde`, `ADEP_mvt` (categorical) and, for the real arm, the 12 ADS-B and 6 stand columns. Hyperparameters, fixed now: learning rate 0.03, 63 leaves, min_child_samples 100,
subsample and colsample 0.8, 600 rounds.

## 2. Arms

**stack** (with the 18 columns), **control_k, k = 1..3** (the 18 columns permuted jointly within airport and day), **stack0** (stage 2 without the 18 columns, reported to show what stage 2 does by itself),
**base** (`p1` alone). Reference, not refitted: the one-stage arm with the same 18 columns, 220.82 (pilot 2).

## 3. Decision (all must hold)

1. stack beats the mean control in each of the two seeds; 2. positive in both months; 3. positive after removing the 100 most favourable rows; 4. airport-day block bootstrap 95 % interval above zero;
5. RMSE gain over the mean control at least **0.3 s**; 6. stack beats base. Reported, not judged: the share of the one-stage gain that the stack keeps, (base - stack) / (base - 220.82), and gain by airport.

## 4. Meaning

Pass: the two-stage form is the architecture for the rebuild (stage 1 trained on every month, stage 2 on the months with tracks). The share kept sets how much a full one-stage retrain on more months can still add.
Fail: ADS-B columns must go into the first stage, and the first stage must be trained on months that have tracks.

## Addendum A, 15 October 2026: pilot 5 passes all six rules

`pilot5_screen.py`, `results/pilot5_screen.json`, log `.cache/research_730/pilot5.log`. Out-of-fold RMSE, mean of two seeds, 339,046 rows: **stack 218.812**, mean control 237.617,
stack0 237.729, base 236.442; one-stage reference with the same 18 columns 220.824. Gain over the controls **+18.81 s**, over the base **+17.63 s**; the stack keeps **113 %** of the one-stage gain.
Both seeds positive (+8,556 / +8,610 MSE per row), January +8,190, July +8,904, +7,947 without the best 100 rows, airport-day bootstrap [+7,755, +9,493]. By airport against the base (MSE per row):
EHAM +16.5k, LSZH +15.6k, EDDM +14.2k, LEBL +13.6k, EDDF +13.1k, LIRF +6.8k, EGLL +5.7k, LEMD +2.7k, LFPG +0.4k, LTFM -0.8k. **Verdict: pass.** A second stage without the ADS-B columns (stack0) is
slightly worse than the base, so the whole gain is the ADS-B evidence. Consequence: the two-stage form is the architecture for the rebuild; stage 1 can be any model trained on all months.
