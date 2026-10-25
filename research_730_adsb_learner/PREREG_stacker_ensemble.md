# Pre-registration: the ADS-B corrector on top of the strong ensemble (pilot 6)

Written 15 October 2026, **before any fit for it**. Same conditions as the earlier pilots. Changes go into dated addenda.

## 0. Why

Pilot 5 showed the ADS-B corrector keeps the whole gain on top of a plain 91-feature LightGBM (236.4 to 218.8). The question for the rebuild is whether it still pays on top of a strong
first stage that already uses neighbour sequences and many layers. `.cache/sequence_release_v47/validate/ensemble_baseline.parquet` holds the held-out predictions (column `baseline`) of the
v45/v47 ensemble for the same 339,046 January and July 2025 main-lane rows, trained on the other ten months: RMSE 216.773, checked 15 October to align row for row with the pilot table and its labels.
These predictions are out of month, so stage 2 can be fitted on them with the same day folds without leaking.

## 1. Design and arms

Exactly pilot 5 (`pilot5_screen.stage2`: residual LightGBM, learning rate 0.03, 63 leaves, min_child_samples 100, subsample and colsample 0.8, 600 rounds, four day-blocked folds, seeds 177 and 277),
with `p1 = baseline` of the ensemble. Arms: **stack** (`p1`, `ersatz_s`, `stunde`, `ADEP_mvt` + the 18 ADS-B and stand columns), **control_k** (the 18 permuted jointly within airport and day, k = 1..3),
**stack0** (without the 18), **base** (`p1`).

## 2. Decision (all must hold)

1. stack beats the mean control in each seed; 2. both months positive; 3. positive without the 100 most favourable rows; 4. airport-day bootstrap 95 % interval above zero;
5. RMSE gain over the mean control at least **0.3 s**; 6. stack beats base. Reported, not judged: gain by airport, and the gain as a share of pilot 5's +17.6 s.

## 3. Meaning

Pass: the rebuild is "strong ensemble + ADS-B corrector", and the size of the gain here is the best estimate of what ADS-B is worth on top of our real model (before the old partial-track layers, which this
replaces). Fail: the ensemble already captures what the ADS-B columns carry, and the earlier gains were the weak base's.

## Addendum A, 15 October 2026: pilot 6 passes all six rules

`pilot6_screen.py`, `results/pilot6_screen.json`, log `.cache/research_730/pilot6.log`. 339,046 rows, mean of two seeds: **stack 200.992**, mean control 217.034, stack0 216.898, ensemble 216.773.
Gain over the controls **+16.04 s**, over the ensemble **+15.78 s**, which is 90 % of pilot 5's gain on the weak base. Both seeds positive (+6,715 / +6,697 MSE per row), January +6,742, July +6,676,
+6,233 without the best 100 rows, airport-day bootstrap [+6,094, +7,371]. By airport against the ensemble (MSE per row): EHAM +12.8k, LSZH +12.1k, EDDF +10.9k, EDDM +10.7k, LEBL +9.6k, EGLL +5.9k,
LIRF +5.5k, LEMD +2.1k, LFPG +1.3k, LTFM -0.1k. **Verdict: pass.** The ensemble already uses neighbour sequences and many layers, yet the ADS-B corrector keeps almost all of its value on top. The rebuild
is therefore "ensemble + ADS-B corrector". Open: the folds are days within the same two months; the cross-month transfer (corrector trained on some months, applied to others) is the next test.
