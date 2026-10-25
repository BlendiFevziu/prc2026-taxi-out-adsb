# Pre-registration: gate-retention columns at the real push-back clock as main-lane columns

Written 4 October 2026, **before any code is run and before any fit**. Nothing below changes after the first fit;
changes go into dated addenda. NM-free: only the challenge's own columns. Private research: no production change, no
candidate, no upload, no merge, no leaderboard use.

## 0. Why, and what is already known

- **Origin.** The public repository of another team (EnioAguiar/prc-taxiout-2026, read 4 October 2026) reports a
  block of "gate retention" columns (`ret_*`) at the real push-back clock as the change that moved its official score
  from 244.89 to 244.18 (v34), with a corrector trained without the rule-served rows worth a further 0.05 s. Their
  text is a lead, not evidence about our model: their base differs from ours.
- **What is already in our 91 features** (printed from `research_250.candidate_features.main_features`, 4 October):
  runway counters around the take-off (`qt_bahn_vor_10`, `qt_bahn_nach_10`, `qt_bahn_vor_30`, `qt_bahn_nach_30`,
  `qt_bahn_luecke_*`, `qt_bahn_ank_vor_10`, `qt_bahn_anteil_60`, `qt_flh_vor_10`, `qt_bahn_rang_10`), arrival context
  (`um_*`, `ank_*`), stand density (`tu_um_dichte_stand_3h`) and the AOBT-3 differences (`dt_aobt_3_*`).
  **Not present:** a count of departures that are past their schedule but not yet pushed at the push-back clock of this
  row, the number of aircraft taxiing at that moment, and the recent push-back delay level of the airport.
- **Earlier evidence in this repository.** Runway configuration columns failed every rule (`research_370_konfig`,
  -0.171 s against controls). Queue counters entered the 91-feature base, so part of what retention measures may
  already be learned.
- **Honest prior: below the 0.3 s gate.** Their gain was measured against their base, which lacked our `qt_*` block.
  The expected gain here is small and the most likely outcome is a fail.

## 1. The seven columns (`research_720_retention/features_retention.py`, tests in `tests/test_features_retention.py`)

All read only `ADEP_mvt`, `RUNWAY_mvt`, `SCHED_TIME_UTC_mvt`, `AOBT_3_flt` and `MVT_TIME_UTC_mvt` of **other**
departures of the same airport and calendar month, never a label of the row itself. For a row i, t is its own
`AOBT_3_flt` (main lane only, so t is always present). A departure j counts only if its own `AOBT_3_flt` is present,
and the row itself never counts.

| column | definition |
|---|---|
| `ret_wait_apt` | departures j at the airport with `SCHED_j <= t` and `AOBT_3_j > t`: past their schedule, not yet pushed |
| `ret_wait_bahn` | the same, only departures on the same runway as i |
| `ret_roll_apt` | departures j with `AOBT_3_j <= t < MVT_j`: pushed and not yet airborne, airport |
| `ret_roll_bahn` | the same, only on the same runway as i |
| `ret_push15`, `ret_push60` | departures j with `AOBT_3_j` in `(t - 900 s, t)` and `(t - 3600 s, t)` |
| `ret_delay15_s` | mean of `AOBT_3_j - SCHED_j` over the departures counted in `ret_push15`, missing if none |

Computed per airport and calendar month (the unit of both the training files and the ranking file, the same choice as
`research_370_konfig`). `ret_wait_*` uses `#{SCHED <= t} - #{max(SCHED, AOBT_3) <= t}` by sorted search, so the row's
own contribution cancels; `ret_roll_*` uses `#{AOBT_3 <= t} - #{MVT <= t}` minus the row itself.

Note on what the columns read: `MVT_TIME_UTC_mvt` and `AOBT_3_flt` of other flights are columns of the challenge's
input files for every row (also the scored ones), so all seven columns are available for the scored rows. The first
hours of a month lack history in training and in 2026 alike.

## 2. Arms and protocol (as `research_370_konfig/PREREG_konfig.md` section 2)

- **real**: the 91 release features plus the seven columns.
- **control 1, 2, 3**: the seven columns permuted **jointly within airport and calendar month**, learning and held-out
  rows with independent seeds (1001 to 1003, 2001 to 2003). Same width as real, row-level link broken.
- **base**: the 91 features alone.

Same three training seeds for every arm (177, 277, 377). One LightGBM, learning rate 0.03, 255 leaves,
min_child_samples 50, subsample and colsample 0.8, **fixed 1,200 rounds**; main lane, target y minus ersatz_s,
prediction max(ersatz_s + model, 300). Fifteen fits, one at a time under the shared heavy-work lock. Runner:
`research_720_retention/retention_screen.py`.

**Precondition, checked before any fit:** `ret_wait_apt` is present (non-missing) on at least 98 % of learning and of
held-out rows, and the feature tests pass. Otherwise the runner stops without fitting.

## 3. Decision

Loss per held-out row = squared error averaged over the three seeds; the control loss is the mean over the three
controls. The columns **pass** only if all six hold:

1. real beats the mean control in **each** of the three seeds;
2. the gain over the mean control is positive in **both** held-out months;
3. it stays positive after removing its **100 most favourable rows**;
4. the 2,000-draw **airport-day block bootstrap** 95 % interval of that gain lies above zero;
5. the RMSE gain over the mean control is at least **0.3 s**;
6. the seed-averaged RMSE of real is below that of the base.

Reported, not judged: gains per airport against control and base, LIRF, LTFM and LFPG pooled, rows with
`ret_wait_apt` above its median against the rest, coverage.

## 4. Stages and what the outcome means

- **Stage 1: pair 1_7** (learning months 2 to 6 and 8 to 12, held out January and July).
- **Stage 2, only if stage 1 passes: pair 3_9**, same arms, seeds and six rules.

Both passing justifies a full-ensemble check under its own registration, not a candidate. Failing at either stage
closes gate-retention columns as main-lane columns.

## 5. Outside this registration

The second lead of the same repository, training the corrector without the rows served by a fixed rule, is not
tested here: our correctors (`research_580_corrector`) are built differently and need their own reading first.

## Addendum A, 4 October 2026: stage 1 (pair 1_7) passes all six rules

`retention_screen.py 1_7`, `results/retention_screen_1_7.json`, log `.cache/research_720/screen_1_7.log`. Coverage:
`ret_wait_apt` present on 100 % of 1,723,531 learning and 339,046 held-out rows. Seed-averaged RMSE: real 223.269, mean
control 223.924, base 223.657. Gain over the mean control **+0.656 s** (every seed positive: +291 / +289 / +300 MSE per
row), over the base **+0.389 s**. January +55, July +487 MSE per row; +132 without the best 100 rows; airport-day
bootstrap 95 % interval [+189, +410]. Per airport against control: LTFM +886, EGLL +557, LIRF +526, LEBL +305, LEMD +165,
LFPG +146, EDDM +126, LSZH +55, EHAM -14, EDDF -34. **Verdict: pass.** The prior in section 0 (below 0.3 s) was too
low. January is weak (+55), July carries most of the gain; stage 2 on pair 3_9 (March and September) tests whether that
is month-specific. Stage 2 runs next under the unchanged six rules.

## Addendum B, 4 October 2026: stage 2 (pair 3_9) passes all six rules

`retention_screen.py 3_9`, `results/retention_screen_3_9.json`. Coverage 100 % on 1,378,679 learning and 344,852 held-out
rows. Seed-averaged RMSE: real 200.820, mean control 201.376, base 201.387. Gain over the mean control **+0.556 s**
(every seed positive: +198 / +260 / +214 MSE per row), over the base **+0.567 s**. March +233, September +215; +100 without
the best 100 rows; airport-day bootstrap 95 % interval [+143, +298]. Per airport against control: EGLL +485, LTFM +411,
LFPG +289, LEBL +288, LIRF +264, EDDM +182, EDDF +124, LSZH +97, EHAM +24, LEMD -5. **Verdict: pass.** Both stages pass, so
under section 4 a full-ensemble check under its own registration is justified. This is not a candidate and not a
production change. The January weakness seen in stage 1 (+55) does not recur here; EDDF and EHAM, negative in stage 1,
are positive in stage 2.
