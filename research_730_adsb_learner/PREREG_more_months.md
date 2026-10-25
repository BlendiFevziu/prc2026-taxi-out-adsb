# Pre-registration: does the ADS-B corrector gain from more training months? (pilot 9)

Written 19 October 2026, **before any fit for it**. Same conditions as the earlier pilots (private research, no leaderboard, no upload, NM-free). Changes go into dated addenda.

## 0. Why

Pilot 7 trained the corrector on one month pair (about 330,000 rows) and applied it to another. We now also hold complete tracks for March, May and December 2025
(December without the 31st, which has no release). Two questions: (A) does a corrector trained on five months beat one trained on two, on the same held-out months; (B) does that
hold in the form the 2026 build uses, where the corrector is applied over the ensemble's predictions. PLAN_adsb_months.md step 4 makes the download of April and August to November
conditional on (A) gaining at least 0.3 s.

## 1. Stage 1

Pairs (1, 7) and (2, 6): the cached leave-pair-out predictions of pilot 7 (`stage1_leave_pair_out.parquet`), unchanged. New group (3, 5, 12): one LightGBM with exactly pilot 7's stage-1 settings
(91 release features, learning rate 0.03, 255 leaves, min_child_samples 50, subsample and colsample 0.8, 1,200 rounds, target `y - ersatz_s`, prediction `max(ersatz_s + model, 300)`), trained on the
main-lane rows of the **nine other** months, seeds 177 and 277. As in pilot 7, the stage 1 of a training month may have seen the held-out pair; only the held-out pair's own stage 1 never saw it.

## 2. Tables

March, May, December 2025: the 12 ADS-B and 6 stand columns with the code of pilot 7 (`features_adsb.compute(stands=True)`, one month's track files at a time, callsign from the raw training file).
Rows of 2025-12-31 are dropped, because their track day is missing for a reason that has nothing to do with the flights.

## 3. Part A, learning curve (stage-1 LightGBM predictions on both sides)

For each held-out pair P in {(1, 7), (2, 6)}, with Q the other pair, the pilot-5 corrector (`pilot5_screen` settings, features `p1`, `ersatz_s`, `stunde`, `ADEP_mvt` + 18 columns), seeds 177 and 277:
- **two**: trained on Q (this is pilot 7's stack arm; its cached predictions are reused);
- **five**: trained on Q + March + May + December.

Judged on the four held-out months together, **five against two**: 1. five beats two in each seed; 2. positive in each of the four months; 3. positive after removing the 100 most favourable rows;
4. airport-day block bootstrap 95 % interval above zero; 5. RMSE gain at least **0.3 s**; 6. five beats the stage-1 base. Pass = more months help; then the remaining 2025 months are fetched.

## 4. Part B, the build's form (applied over the ensemble)

Held out: January and July 2025, with p1 = the ensemble's held-out predictions (`.cache/sequence_release_v47/validate/ensemble_baseline.parquet`, RMSE 216.773, the base of pilot 6 and of the 2026 build).
- **two_ens**: corrector trained on February + June (stage-1 LightGBM p1);
- **five_ens**: trained on February + June + March + May + December (stage-1 LightGBM p1).

Same six rules, five_ens against two_ens, rule 2 over January and July. Reported, not judged: both against pilot 6's stack (**200.992**, the within-month corrector trained on the ensemble's own
predictions, which is the design of the current 2026 file).

## 5. Meaning for the 2026 file

If Part B passes **and** five_ens is below 200.992, the 2026 file is rebuilt with the corrector trained on all seven months held (January and July with their leave-pair-out stage-1 p1, so that every
training row has a held-out p1 of the same kind). Otherwise the current file (SHA-256 `acca1ea3…`) stays. Nothing here is tuned; the settings are those of pilots 5 and 7.

## Addendum A, 19 October 2026: both parts pass; the rebuild condition does not

`pilot9_screen.py`, `results/pilot9_screen.json`, log `.cache/research_730/pilot9.log`. Stage 1 for March, May, December (trained on the nine other months): 194.874 / 194.817. Table: 506,385 rows (March 163,367,
May 183,597, December 159,421 after dropping the 31st), matched 78.4 / 78.9 / 84.3 %.
**Part A** (four held-out months, 662,308 rows): two 205.041, **five 203.566 (+1.48 s)**; both seeds positive (+630 / +575 MSE per row), every month positive (January +801, February +37, June +771, July +710),
+433 without the best 100 rows, bootstrap [+440, +749]. Pass.
**Part B** (January and July over the ensemble, 339,046 rows): two_ens 203.893, **five_ens 202.166 (+1.73 s)**; seeds +718 / +685, January +746, July +665, +424 without the best 100, bootstrap [+532, +868]. Pass.
**But five_ens 202.166 is above pilot 6's 200.992**, so section 5's rebuild condition fails and the current 2026 file stays. Reading: more months help, and a corrector that learns over the stage-1 LightGBM's
predictions loses about 1.2 s against one that learns over the ensemble's own. Part A's pass meets PLAN_adsb_months.md step 4, so April and August to November are being fetched (started 19 October 20:21).
