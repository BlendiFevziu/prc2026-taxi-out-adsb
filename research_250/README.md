# Post-v44 research: 240 milestone, 230 stretch target, first-place goal

The official best is now **268.2394 s (v46)**: v45's main lane plus the research_240 fallback
anchor, scored 13 September. v45 scored **269.4753 s**, a **2.8587 s** improvement from v44.
The timestamp-corrected/runway-traffic candidate completed all-year training and
export and was submitted once on 12 September. Original v44 weights and legacy
feature definitions remain unchanged; v45 uses the explicit 91-column versioned schema.
Complete January/July ensemble validation gained 2.313097 s overall
(309.577417 -> 307.264321). See [full validation](results/full_ensemble_gate_1_7.json),
[official result](../results/upbeat-goblin_v45.json) and [release handoff](RELEASE_RUN.md).
Earlier "next steps" below are historical motivation, not outstanding release work.
On 12 September the user changed the next major milestone to **240 s**, the stretch
target to **230 s**, and the competitive goal to **first place**. Relative to the
recorded official 268.2394 s (v46), these require 28.2394 s less RMSE (19.95% less MSE)
and 38.2394 s less RMSE (26.48% less MSE), respectively. These are research objectives, not
performance forecasts; the local validation gain cannot be subtracted from the
official score as a guaranteed result. First place requires a fresh leaderboard
check. The old 270.9373 checkpoint is passed, not the stopping objective. The directory name
and historical 250-target analyses below are retained for reproducibility. Do not
change the frozen release recipe or start competing fits because the goal changed.

## How the team-profile findings inform the next research phase

The profiles we checked provide research leads, not disclosed winning PRC recipes.
The earlier public [ASHRAE first-place solution](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/tree/master/solutions/rank-1),
coauthored by a Matthew Motoki, combines familiar model families with different
target transformations and training partitions. The identity match to the PRC
participant is unconfirmed, and this is not evidence of the current leader's method.
Its transferable hypothesis is complementary predictions from different target and
training views, not merely more seeds of the same residual model. Competition-specific
leaked-data practices from another challenge are not transferable or authorized.

The [Junguo Song team profile](https://ansperformance.eu/study/data-challenge/dc2026/teams/enthusiastic-daisy.html)
and [Jiawei Jing team profile](https://ansperformance.eu/study/data-challenge/dc2026/teams/upstanding-firefly.html)
suggest trajectory/time-series expertise, but do not establish what those teams
actually submitted. They support investigating richer, ordered nearby-flight context
rather than assuming a larger version of our existing flat neural network will help.
Our own error audit separately motivates further work on unlinked flights and extreme
reporting errors. Previous failed fallback gates, target caps and larger flat networks
remain negative evidence; consult the experiment log before repeating them.

For the next research phase, assess these hypotheses with paired independent
temporal validation, complementary-error checks and full, untrimmed RMSE. Change the
representation or architecture when measured evidence supports it; a wholesale rewrite
is not itself an improvement. A bounded research pass should return its results,
negative findings and next discriminating test even if neither target is reached.
Failure of one pass is not proof that 230 is impossible, and aspiration is not proof
that it is achievable. No new experiment or upload is launched by this goal update.

## Concrete pipeline defect

The supplied movement timestamps load as `datetime64[us, UTC]`. Casting to `int64`
retains microseconds. Several builders divide that number by 1e9 as though it were
nanoseconds. On the observed input this changes:

| Feature | Intended definition | Existing v44 behavior |
|---|---|---|
| `auslast` | Departures within ±30 minutes | Approximately ±20.83 days |
| `um_dichte_stand_3h` | Same-stand movements within ±90 minutes | Approximately ±62.5 days |
| Two flight-ID time gaps | Seconds | Values 1,000 times too small |
| Two flight-ID rates | ID difference divided by seconds, floor 1 s | The wrong gap frequently hits the denominator floor |

`features_time_units.py` builds corrected values after explicitly normalizing datetime
units. New cached columns have `tu_` prefixes, so scored weights never silently receive
changed definitions. Main-model experiments compare replacing the six affected
features with adding corrected versions alongside the originals. The fallback test
adds only corrected `tu_auslast` to its existing 35 features.

The inactive `features_queue.py` and historical `fid_gap_cv.py` also contain the
unit assumption. Pandas timedelta `.dt.total_seconds()` calculations are unaffected.
Previous queue rejection therefore needs a properly timed rerun before drawing a
strong conclusion about the whole feature family.

## January/July exploratory screen

All feature references are built independently on the ten learning source files and
two held-out source files. Raw absolute IDs are never predictors. There are 17,097
training and 5,373 validation fallback records. Target encoding is fitted outside
the outer holdout, with training stand statistics calculated out of month.

| Single-LightGBM main component | RMSE (s) | Gain vs v44 component |
|---|---:|---:|
| v44 features | 226.554 | — |
| Replace six affected features | 225.897 | +0.657 |
| Add corrected features, retain old ones | 226.495 | +0.059 |
| Add corrected features and 12 reporting features | 226.239 | +0.315 |

Same seed 177, maximum 1,400 trees, learning rate 0.03, 255 leaves, minimum leaf 50,
80-round early stopping and 300-second prediction floor. This does not include the
CatBoost/neural/airport blend or the main reporting classifier. January/July is the
exploratory split used to choose the replacement candidate; later pairs are confirmation.

| Fallback component | RMSE (s) | Outcome on this holdout |
|---|---:|---|
| Rebuilt v44 | 1759.277 | Reference |
| Add corrected traffic count | 1755.557 | Small gain; send to remaining folds |
| Corrected count + all reporting context | 1813.111 | Worse |
| Two-part capped/excess conditional expectation | 2424.911 | Worse |
| Inner-cross-fitted squared-error gate | 1799.736 | Worse |
| Half context prediction, half learned gate | 1785.454 | Worse |
| Corrected count + two missingness features | 1800.578 | Worse |
| Corrected count + four record-time anomalies | 1766.505 | Worse |
| Corrected count + record-time + missingness | 1783.067 | Worse |

The fallback baseline retains v44's 600-tree classifier, 400-tree normal regressor,
300-tree extreme-height regressor with minimum leaf 10, probability power 2.5, height
model weight 0.3, and a zero prediction floor. It is not the older cached fallback.
The expected-value and gate alternatives were actually fitted; their motivation alone
is no evidence of improvement. They remain excluded.

The five worst fallback rows account for 68.29% of its squared error on this holdout.
Two LFPG records alone contribute about 62%; the labels are 84,240 and 58,206 seconds
against predictions near 715 seconds. Do not remove these from the primary metric.
Historical full-year fallback arrays inspected before this screen were pre-v44,
not v44 out-of-fold predictions.

## Validation status and safeguards

Fallback confirmation is complete. Adding the corrected count reduced pooled RMSE
from **1508.393 to 1504.072 seconds** across 22,470 held-out rows, with four positive
and two negative month pairs. Its MSE gain is 13,016.983; the descriptive airport-day
bootstrap interval is [-33,730.506, 68,395.408], including zero. This is weak evidence,
so the fallback remains unchanged. Full details: [fallback results](results/fallback_six_pair.json).

| Held-out source months | v44 fallback RMSE | Corrected count added |
|---|---:|---:|
| January/July | 1759.277 | 1755.557 |
| February/August | 1365.853 | 1340.573 |
| March/September | 1104.678 | 1096.498 |
| April/October | 784.020 | 775.800 |
| May/November | 1912.550 | 1919.921 |
| June/December | 1633.538 | 1635.873 |

Main-component confirmation is complete: **209.808 → 209.133 seconds pooled RMSE**
on 2,062,577 held-out predictions, a **0.674-second** gain, positive in all six pairs.
The MSE reduction is 282.497, with a descriptive airport-day bootstrap interval of
[230.701, 338.541]. Nine of ten airports improve in pooled MSE; Madrid is slightly
worse. These are single-LightGBM component results, not new leaderboard scores.
Bootstrap intervals describe the observed year and exclude model-selection and
future-year uncertainty. See [all six-pair results](results/time_units_six_pair.json).

| Held-out source months | v44 main component | Corrected units | RMSE gain |
|---|---:|---:|---:|
| January/July | 226.554 | 225.897 | 0.657 |
| February/August | 219.790 | 219.173 | 0.617 |
| March/September | 203.365 | 202.639 | 0.726 |
| April/October | 201.441 | 200.986 | 0.455 |
| May/November | 196.893 | 196.776 | 0.117 |
| June/December | 210.224 | 208.777 | 1.447 |

Replacing the faulty fallback count, instead of adding its correction, was checked
as well: pooled RMSE worsened from 1508.393 to 1513.648, with three positive and three
negative pairs. See [replacement results](results/fallback_replacement_six_pairs.json).
The correction should therefore be a main-branch candidate, not a global change
silently applied to every model input.

A separate architecture screen averaged the original fallback fit and five fixed
bootstrap replicates, with equal weights and no new features. It worsened January/
July RMSE from 1759.277 to 1825.919, failing the pre-specified 100,000 MSE-gain bar.
It did not advance to further folds. The stand encodings remain trained out of month;
the bootstrap varies learners conditional on those encodings. See
[bagging results](results/fallback_bagging_screen.json).

## Correctly timed runway traffic: confirmed component improvement

The ten `qt_` traffic/spacing features passed the pre-specified January/July
advancement bar of 0.3 seconds, improving the corrected baseline by 2.199 seconds.
The same seed, feature recipe and 1,400-tree budget were then used on the other five
source pairs. Every pair improved.

| Held-out source months | Corrected units | Corrected units + runway traffic | Incremental gain |
|---|---:|---:|---:|
| January/July | 225.897 | 223.698 | 2.199 |
| February/August | 219.173 | 216.904 | 2.269 |
| March/September | 202.639 | 200.087 | 2.552 |
| April/October | 200.986 | 198.332 | 2.654 |
| May/November | 196.776 | 194.213 | 2.563 |
| June/December | 208.777 | 206.335 | 2.442 |

Across **2,062,577 held-out flights**, pooled main-component RMSE is
**209.808 → 206.694**, a **3.113-second gain** versus the v44-feature single LightGBM.
The runway layer contributes 2.439 seconds beyond the unit-only correction. The
five confirmation pairs alone improve from 205.675 to 203.182 (2.493 seconds).
These are row-weighted pooled scores, not an average of fold RMSEs.

The combined candidate improves all ten airports. Incremental queue features improve
nine; Rome loses 21.747 MSE incrementally but remains 144.205 MSE better than the v44
component after combining both changes. The descriptive airport-day 95% MSE-gain
interval is [1,226.580, 1,372.850] for the combined change and [951.226, 1,076.828]
for the incremental queue layer.

The improvement is not solely an extreme-row effect: the baseline-worst 1% improve
by 32,802.829 MSE within that subset, and the other 99% improve by 978.459 MSE.
All rows remain in the primary metric. Bootstrap intervals exclude model-selection
and future-year uncertainty; early stopping still observes each validation fold,
as it does in the baseline. Full metrics and per-fold code hashes are in
[the six-pair queue record](results/queue_six_pair.json).

No CatBoost/neural/airport blend or main reporting classifier was fitted in these
screens. The result is not a full-ensemble score or a leaderboard forecast.

## Same-flight arrival relinking

A separate label-free rule looked for a unique arrival 10 minutes to 8 hours after
the departure, matching departure airport, destination, callsign and aircraft type.
It recovered an off-block anchor for 775 of 22,470 fallback records. Known-anchor
control records mostly agreed, but there were occasional wrong matches. None of the
ten worst-error records in any fold was covered. Direct anchor replacement worsened
five of six pairs and pooled RMSE (1508.393 → 1510.148).

Even impossible perfect predictions on the matched subset would remove only
11,664.813 fallback MSE on these data, far below the roughly 760,463 fallback MSE
reduction needed if that branch alone were to carry the target of 250 at the recorded
scoring density. This is a limit of this matching rule on the observed examples,
not an information bound on richer linkage methods. See
[the complete probe](results/arrival_relink_six_pairs.json).

Source-file start months define folds, as in the v44 experiments. Two departures in
the March and September files have movement timestamps three seconds before those
months begin. A timestamp-month-disjointness assertion initially stopped February/
August before fitting. Source inspection found no shared record IDs. The guard now
checks record independence, and core-cache sidecars record actual source-file months.
Do not infer a cache's source split from its movement timestamps.

An initial reporting-ablation invocation failed before fitting because relative paths
were resolved after changing directory. Paths are now resolved first. A metadata-only
fix also records `tu_auslast` in the expanded candidate list; actual model inputs and
the reported screen scores were unchanged.

Twenty lightweight regression checks pass, including timestamp-unit invariance,
known clock-window counts, airport isolation, label independence and preserved v44
feature lists. No target-bearing departure field is read by any new feature builder.

## Reproduction and next steps

From the repository root, use a local ignored cache. Model fits must stay sequential.

```bash
python3 -B research_250/fallback250_cv.py --project . --output .cache/rmse250 --pairs '1,7' --main-screen
python3 -B research_250/fallback250_cv.py --project . --output .cache/rmse250 --pairs '2,8;3,9;4,10;5,11;6,12' --variants ''
python3 -u -B research_250/main_time_cv.py --project . --core-dir .cache/rmse250 --output .cache/rmse250/main-confirmation
python3 -B research_250/summarize250.py --results .cache/rmse250 --main-results .cache/rmse250/main-confirmation --output .cache/rmse250/aggregate.json
python3 -u -B research_250/run_queue_confirmation.py --project . --core-dir .cache/rmse250 --baseline-dir .cache/rmse250 --output .cache/rmse250/queue-screen --pairs '1,7'
python3 -u -B research_250/run_queue_confirmation.py --project . --core-dir .cache/rmse250 --baseline-dir .cache/rmse250/main-confirmation --output .cache/rmse250/queue-confirmation
python3 -B research_250/summarize_queue.py --screen-dir .cache/rmse250/queue-screen --confirmation-dir .cache/rmse250/queue-confirmation --output .cache/rmse250/queue-summary.json
python3 -B -m unittest discover -s tests -v
```

The supplemental `reporting_ablation.py` takes explicit `--core-train` and
`--core-valid` parquet paths from the cache manifests. It reproduces the narrowed
missingness and inverse-record-time screens without rebuilding raw features.

The corrected main and runway-traffic features passed the six-pair single-LightGBM gate
and the subsequent complete January/July ensemble check; v45 is now officially scored.
The first screen's 0.3-second advancement bar was
exceeded; the five subsequent comparisons kept the configuration unchanged.
The large fallback reporting errors remain unresolved; do not promise that the
timestamp correction alone supplies the remaining 29.4753-second gap to 240.

### Safe ensemble integration

`candidate_features.py` prepares an explicit 91-column main schema: six `tu_`
replacement names and ten `qt_` runway-traffic columns. It leaves every original
column and the 35-column fallback list unchanged. On all 344,419 January/July
departures, `check_candidate_replay.py` reproduced the experimental model inputs
exactly from 688,418 raw movements. See [replay checks](results/candidate_replay_1_7.json).
These are traffic counts and spacing features, not direct observations of a physical
taxi queue. The model still predicts the reporting residual against the AOBT anchor.

`ranking_schema_preflight.py` also successfully built all 91 main columns on the
actual 344,841 ranking departures (339,551 main and 5,290 fallback). Their movement
IDs match the submission template, the departure targets remain unobserved, and no
new feature contains infinity. Missing flight IDs remain supported as missing
features. This check loaded no model weights and generated no predictions or
submission; see [ranking-input preflight](results/ranking_schema_preflight.json).

The following pre-release checklist is historical: these gates were completed by
the v45 release. Do not restart them merely because they remain documented:

1. Validate paired full ensembles with independently built source-file references.
   Retrain the main LightGBM, CatBoost, neural, airport-specific and reporting-gate
   components with the explicit candidate main list; keep fallback inputs unchanged.
   Retain the current blending and submission postprocessing for the comparison.
2. Recheck training length for the new feature representation using training-only
   inner validation. The exploratory queue fit selected 1,179 LightGBM iterations,
   versus the current production recipe's fixed 890; four confirmation fits selected
   1,387–1,400 iterations and the other selected 1,349. Blindly reusing 890 could
   undertrain it. Do not select iteration counts or blend weights on ranking feedback.
3. Report untrimmed paired MSE/RMSE, airport effects and baseline-defined tail
   diagnostics. A single-LightGBM gain is not a measured full-ensemble gain.
4. Only after that gate, train a separately named bundle on all twelve 2025 months
   and add version-aware feature construction to its inference path. The current
   unmodified submission entry point does not build the prefixed candidate columns.
5. Verify the new submission locally, preserve scored artifacts and upload only
   when the user authorizes it. Do not silently replace `models_fid_arrivals.pkl`.

Avoid `framework.daten()` for these paired experiments: its historical workflow
builds all-month references before splitting and removes older feature caches.
The isolated research drivers instead use independent source-file references and
retain their output records. The v44 model and submission SHA-256 values were
rechecked against the scored manifest and still match exactly.
