# Assigned-stand departure detector: frozen rules and validation plan

Written 23 September 2026 on branch `claude/munich-stand-detector`, **before any validation date was evaluated**.
Nothing in sections 1 to 4 may change after the first validation run; changes go into a dated addendum.
Private research: no production change, no candidate, no upload, no merge.

## 0. Where this comes from

Codex's diagnostic on **13 December 2025** at Munich (`private-nm-archive/full-month-research/
CLAUDE_LUIS_REVIEW_2026-09-23.md`) found ground points near the recorded block time for most matched departures,
while the old detector (`adsb-lol/offblock_from_tracks.py`) was late. Its recommendation: an assigned-stand
detector that rejects later taxiway stops, frozen before evaluating new labelled dates.

**Gate 0, passed.** Munich has ADS-B ground coverage on the scored dates: in samples of 6 and 13 July 2026,
88 of 106 and 47 of 60 Munich aircraft show ground points, against 7 of 408 on 13 July 2025
(`research_290/adsb_coverage_gate.py`). Coverage moves over time: Rome went from 352 of 412 to 0.

## 1. The frozen detector (`research_290/stand_detector.py`, `detect_onset`)

- **Stand.** `STAND_mvt` mapped to OSM `parking_position` vertices with `features_geometry.stand_kandidaten`.
  A bare number with no exact key maps to the union of its lettered sub-positions (111 -> 111A + 111B).
  Candidates whose centres lie more than **100 m** apart make the stand **ambiguous**: abstain.
- **Run.** The final run of consecutive points within **R_STAND = 80 m** of the stand before take-off
  (plus 60 s margin), plus the first point after it.
- **Transitions.** A point j is a parked-to-motion transition when point j-1 is stationary (**below 2.0 kt**)
  and within **R_PARK = 30 m** of a stand vertex, and point j is moving (**2.0 kt or faster**) or outside 80 m.
- **Estimate.** The **first** transition; off-block = **midpoint** of (t_{j-1}, t_j].
- **Abstain** when: no points, no point at the stand, no transition, the bracket exceeds **120 s**, or the first
  and last transitions are more than **600 s** apart (a tow or reposition inside the stand area).
- Matching is the old pipeline's, unchanged: callsign candidates and visit end within 300 s of take-off.

## 2. How the rule was reached, all on the development day only

Honesty matters here because the rule changed three times, all on 13 December:

1. The pre-listed estimates of `detect` (last rest at the stand, first leave, their midpoint) **all failed**:
   median **+257, +285, +271 s**. The tracks showed why: push-back runs at 2 to 4 kt, below the old 4 kt
   threshold, and ends in a stop 30 to 70 m from the stand, still inside the radius; taxi follows minutes later.
2. First motion from the parked position fixed it. The parked distance was measured (median 15 m before the
   block, 41 m for stops after it) and a grid of 25 or 30 m x three estimates was fixed before running;
   the criterion (most within 60 s, then lowest MAE) chose **30 m, midpoint**.
3. Mapping bare stand numbers to lettered sub-positions raised mapped stands from 165 to 267 of 308, but brought
   two misses of **-3,410 and -3,206 s**. Both had first and last transitions an hour apart; every other row
   spread at most 363 s. The **600 s spread rule** removes exactly those two (300 and 900 s give the same).

**Development result of the frozen rule:** 359 departures, 308 uniquely matched, **90 estimates**; error against
the recorded block median **+17.3 s**, MAE **34.4 s**, RMSE **62.4 s**, **81 of 90 within 60 s**. The old detector
on the same day: 149 estimates, median **+262 s**, 17 within 60 s.

**Replication of Codex's diagnostic is close but not exact**: 308 matched (Codex 344), 213 with a ground point
within 60 s of block (234). **The old detector does not reproduce** (149 detections at +262 s against Codex's
55 at +635 s): Codex's "old detector" was a different version. Reported, not reconciled.

## 3. Validation

**New labelled dates, never inspected by anyone: 3, 10, 17 and 20 December 2025** (Wednesdays and one Saturday),
Munich only, same adsb.lol prod-0 source. The validation script reads their labels only after this document is
committed.

**Comparator.** No saved out-of-sample model exists for December: every production model trained on it. So one
single LightGBM is fitted with the repository's screen settings (1400 rounds, lr 0.03, 255 leaves,
min_child_samples 50, subsample and colsample 0.8, seed 177) on months **1 to 11** by strict source month, main
lane, the 91 release features, target `y - ersatz_s`, prediction `max(ersatz_s + model, 300)`, with **a fixed
1,200 rounds and no early stopping**, because stopping on December would read its labels. It is **weaker than
the production ensemble** (on January/July 2025, 223.70 against 216.77), so any detector advantage measured
against it is **overstated** for production. The naive NM taxi `ersatz_s` is reported as a second reference.

**Rows compared.** Munich main-lane departures on the validation dates where the detector gives an estimate;
the detector's taxi time is `MVT - estimate`. Fallback rows are reported separately.

## 4. Decision rules, fixed now

The detector justifies further work toward a candidate only if **all** hold on the pooled validation dates:

- **V1 accuracy transfers.** Error against the recorded block: MAE at most **60 s** and at least **75 %** within 60 s.
- **V2 it beats the comparator on identical rows**, on **every one** of the four dates.
- **V3 not concentrated.** The squared-error gain stays positive after removing the **5 %** most favourable rows.
- **V4 interval.** A 2,000-draw bootstrap over flights gives a 95 % interval of the gain above zero.

Reported but not gating: coverage per date, the gain per covered row, what that implies on the official scale,
fallback rows, and the error distribution's tail. **Passing does not make a candidate**: it would still need the
full ensemble comparison on January and July, and a coverage check on the scored January 2026 dates.

## Addendum A, 24 September 2026: the validation result, and why it failed

Run once on 24 September at 08:17 UTC, at commit 725de04 with a clean tree: `research_290/munich_validate.py`,
output `research_290/results/munich_validate.json`. Sections 1 to 4 were not changed.

**Comparator** (`research_290/december_model.py`, 222 s, `results/december_model.json`): 1,898,549 training rows
from source months 1 to 11, 164,028 December main-lane rows, IDs disjoint, replay exact. December main-lane RMSE
197.54 s (the NM taxi `ersatz_s` alone 374.60 s); on Munich's 11,905 December rows 199.29 s.

**Coverage.** 1,626 departures, 1,434 uniquely matched, **447 estimates** (27.5 % of departures; 117, 106, 108 and
116 per date). Abstentions: no parked transition 596, no point at the stand 154, stand unmapped 152, bracket over
120 s 70, transitions spread over 600 s 14, no points 1. One estimate fell on a fallback row.

| Check | Measured | Rule | Outcome |
|---|---|---|---|
| V1 accuracy | MAE 47.2 s; 392 of 447 (87.7 %) within 60 s | MAE at most 60 s, at least 75 % | pass |
| V2 every date | detector against comparator RMSE: 302.8 vs 161.9 (3 Dec), 73.3 vs 280.4, 89.0 vs 204.5, 53.5 vs 265.7 | lower on all four | **fail** (3 Dec) |
| V3 not concentrated | MSE gain -9,278 per row after removing the most favourable 5 % | above 0 | **fail** |
| V4 interval | flight bootstrap 95 % [-33,885, +73,347] | above 0 | **fail** |

Pooled over the 446 identical main-lane rows: detector RMSE 167.29 s, comparator 232.17 s, `ersatz_s` 545.01 s,
gain 25,917 MSE per row. Error against the recorded block: median +26.9 s, RMSE 167.1 s, 428 within 120 s,
extremes -3,178.9 and +683.5 s.

**Verdict, as fixed in section 4: the frozen detector does not justify further work toward a candidate.**

### Why it failed: post-hoc, description only

Computed after the verdict with `research_290/munich_validate_posthoc.py` (`results/munich_validate_posthoc.json`).
It changes no rule and no outcome above.

**One row decides all three failed checks.** On 3 December at stand 215 (a bare number mapped to its lettered
sub-positions) the estimate was 3,179 s early, which is **81.0 % of the detector's squared error**. Without that
row, over 445 rows, the detector scores 73.1 s against the comparator's 232.3 s, a gain of 48,637 per row. The
track shows a 53-minute turnaround in which the aircraft never left the 80 m radius, so the final run began with
the arrival. Docking produced a still-to-moving step (1.4 kt at 23.8 m, then 2.8 kt), and within the 600 s before
it the aircraft had been at 142 kt: the landing roll. It then stood 36.8 m from the nearest stand vertex, beyond
R_PARK, and was moving 27 s after the recorded block, so the real push-back never counted as a transition and the
spread rule had nothing to compare.

The other 19 misses beyond 120 s are milder, 17 late and 2 early. Eleven of the 16 late misses between +127 and
+357 s lie on stands 181 to 189. The two traced (184 and 185 on 3 December) have two stand vertices about 55 m
apart; neither has an ADS-B point in the 15 minutes before the recorded block, the first one after it is already
moving, the push-back ends 29 to 36 m from a vertex, and the chosen transition is the taxi start. One +684 s row is a recorded block that precedes the movement (the
aircraft stood 9 m from the stand for about ten minutes after it), and one -529 s row shows speed and position
jumping around the 2 kt threshold while parked.

Every estimate classified by the signature of its chosen transition (rules in the script, 30 kt and 600 s):

| Class | Within 60 s | 60 to 120 s | 120 to 600 s | Over 600 s |
|---|---|---|---|---|
| Arrival: faster than 30 kt within 600 s before | 0 | 0 | 0 | 1 |
| Push-back unobserved: motion at the stand, then a parked stretch under 600 s | 20 | 15 | 9 | 0 |
| Several transitions | 43 | 6 | 1 | 0 |
| Clean | 329 | 15 | 7 | 1 |

On the development day the arrival class is empty (71 clean, 6 push-back unobserved, 13 several transitions). The
clean class alone scores MAE 33.3 s and RMSE 60.7 s on the validation dates.

**A model-disagreement gate was considered and rejected.** Four rows disagree with the comparator by more than
900 s. The detector is closer on three of them: taxi-outs of 3,118 to 3,476 s that the comparator underestimated
by 1,035 to 2,443 s. Only the catastrophe favours the model. The gate would give up 10.47 million s² to save
10.08 million. On these dates the detector was most valuable exactly where the model failed worst.

## Addendum B, 24 September 2026: v2, one added rule, four fresh dates

Written after addendum A, and committed and pushed **before any data of the dates below was downloaded**. v1's
verdict stands. This is a second, separately registered attempt and must be read as the second of two.

**The one change** (`stand_detector.detect_onset_v2`). v1's `detect_onset` runs unchanged. An estimate it returns
is withdrawn (status `arrival_transition`) when any observation in [t_before - 600 s, t_after) is faster than
30 kt. A push-back follows a parked turnaround, and nothing parked was at landing or fast-taxi speed ten minutes
earlier, so such a transition is a stop-and-creep step while docking. The two thresholds are those of the post-hoc
classification in addendum A and were not varied. On the five used dates the rule removes exactly one estimate, the
-3,178.9 s failure, and nothing else (`research_290/v2_used_dates_check.py`). Every other constant, the stand
mapping, the matching and the estimate are v1's.

**Considered and not adopted.** The model-disagreement gate (addendum A). A rule against push-backs seen only in
motion, which on the used dates would remove 20 estimates within 60 s to fix 9 misses of 2 to 10 minutes.

**Fresh validation dates, never opened by anyone: 5, 8, 11 and 14 December 2025** (Friday, Monday, Thursday,
Sunday). Munich, adsb.lol prod-0. Chosen before any download, spread over the first half of the month, away from
the holidays and from the five used dates.

**Unchanged:** the comparator predictions (`december_model.parquet`, fitted before either validation; no row of
these dates has been looked at), the decision rules V1 to V4 of section 4 with their thresholds, the bootstrap seed
and the 5 % trimming. V1 is judged on all v2 estimates, V2 to V4 on identical main-lane rows. Script:
`research_290/munich_validate_v2.py`, which passes the v2 detector, these dates and new output names to
`munich_validate.main`. That function was parametrised for this, and the v1 run through it reproduces its committed
output exactly (JSON unchanged, all 1,434 rows identical).

**What the outcome would mean.** Passing justifies building toward a candidate, nothing more. Open before any
submission: a production-strength comparison (the comparator is a single LightGBM, about 7 s weaker than production
on January and July 2025), detector coverage on the scored January 2026 dates, and the realised share on the
scored July 2026 dates (21 to 23 % of matched departures in the samples, against 27.5 % in December). Failing
closes the Munich December line: a third variant would be chosen after two looks at the same month.

## Addendum C, 24 September 2026: the v2 result

Run once on 24 September at 09:53 UTC, at commit ffccea1 with a clean tree: `research_290/munich_validate_v2.py`,
output `research_290/results/munich_validate_v2.json`. Addendum B was not changed.

**Coverage.** 1,688 departures, 1,496 uniquely matched, **498 estimates** (29.5 % of departures; 129, 124, 132 and
113 per date). Abstentions: no parked transition 594, no point at the stand 168, stand unmapped 148, bracket over
120 s 74, transitions spread over 600 s 11, **arrival transition 3**. Two estimates fell on fallback rows.

| Check | Measured | Outcome |
|---|---|---|
| V1 accuracy | median +23.8 s, MAE 36.7 s, RMSE 61.3 s; 443 of 498 (89.0 %) within 60 s, 481 within 120 s; extremes -333.6 and +468.7 s | pass |
| V2 every date | detector against comparator RMSE: 41.2 vs 206.1 (5 Dec), 64.3 vs 117.1 (8 Dec), 67.4 vs 146.6 (11 Dec), 62.0 vs 149.7 (14 Dec) | pass |
| V3 not concentrated | +11,666 MSE per row after removing the most favourable 5 % | pass |
| V4 interval | flight bootstrap 95 % [+17,080, +26,740] | pass |

Pooled over the 496 identical main-lane rows: detector RMSE **59.55 s**, comparator 158.64 s, `ersatz_s` 447.21 s,
gain 21,620 MSE per row. A half blend of detector and comparator (descriptive) scores 87.30 s, worse than the
detector alone.

**Verdict, as fixed in addendum B: v2 justifies building toward a candidate, and nothing more.** It is the second of
two attempts on Munich December data, and the rule it adds was found on the first attempt's validation dates.

**Post-hoc** (`research_290/v2_withdrawn_rows.py`, after the verdict). The three withdrawn estimates would have been
-3,224.8, -3,089.7 and -3,259.0 s under v1, on 8, 11 and 14 December: the failure of addendum A again, about once
a day, and each one a main-lane row. With them, v1 would have failed V2 on all three of those dates (295.5 vs 116.7,
276.2 vs 146.6 and 312.7 vs 149.0 s). The rule removed no estimate that v1 had right.

## Addendum D, 24 September 2026: production-strength test at Munich, January 2025

Written after addendum C and after the transfer test (`PREREG_transfer_2025.md`), and committed and pushed before
three of the four dates below were downloaded. The owner approved this step on 24 September.

**Why.** v2 passed at Munich against one LightGBM, about 7 s weaker than production. January 2025 is the only month
in which Munich had ADS-B ground coverage (394 of 430 aircraft on 16 January, against 7 of 408 on 13 July 2025) and
production-strength held-out predictions exist: those of the full v45 ensemble (`ensemble_baseline.parquet`,
SHA-256 checked, trained without January and July 2025, 196.95 s on January's main lane).

**Fixed.** Detector v2 unchanged (ffccea1). Dates: **8, 16, 24 and 26 January 2025** (Wednesday, Thursday, Friday,
Sunday). 16 January was downloaded for the coverage check and run through the detector without labels (27.0 % of
departures got an estimate); no error was computed on it. The other three are downloaded after this commit.
Decision rules V1 to V4 of section 4, unchanged. Script: `research_290/munich_validate_jan2025.py`, which passes
the v2 detector, these dates, this comparator and new output names to `munich_validate.main`. That function now
also takes a comparator; the v1 and v2 runs through it reproduce their committed outputs byte for byte.

**What the outcome means.** A pass justifies a Munich-only candidate. Building it needs the tracks of all 62 scored
days (about 215 GB) and the owner's approval, both before that download and before any upload. A fail ends the
Munich line: where the detector is accurate, it would then not beat production.

## Addendum E, 24 September 2026: the production-strength result at Munich

Run once on 24 September at 14:49 UTC at commit ec63c23: `research_290/munich_validate_jan2025.py`, output
`research_290/results/munich_validate_jan2025.json`. Addendum D was not changed. (Two earlier attempts that day
stopped before the run because github.com was unreachable; the guard ran the test only once all four days' tracks
existed.)

**Coverage.** 1,499 departures, 1,448 uniquely matched, **454 estimates** (30.2 %; 135, 103, 104 and 112 per
date). The arrival rule withdrew one estimate. One estimate fell on a fallback row.

| Check | Measured | Outcome |
|---|---|---|
| V1 accuracy | median +23.2 s, MAE 36.6 s, RMSE 65.5 s; 405 of 454 (89.2 %) within 60 s, 437 within 120 s; extremes -504.0 and +435.4 s | pass |
| V2 every date | detector against the v45 ensemble: 73.6 vs 215.8 (8 Jan), 45.1 vs 127.0 (16 Jan), 72.7 vs 115.1 (24 Jan), 64.4 vs 101.6 (26 Jan) | pass |
| V3 not concentrated | +6,527 MSE per row after removing the most favourable 5 % | pass |
| V4 interval | flight bootstrap 95 % [+11,834, +27,080] | pass |

Pooled over the 453 identical main-lane rows: detector RMSE **65.60 s**, v45 full ensemble **151.81 s**, `ersatz_s`
372.64 s; gain 18,744 MSE per row, close to the 21,620 measured against one LightGBM in December. A half blend
(descriptive) scores 83.36 s. Official-scale upper bound at this coverage, Munich only: **0.80 s**.

**Verdict, as fixed in addendum D: a Munich-only candidate is justified.** Building it needs the tracks of all 62
scored days and the owner's approval, before that download and before any upload. The bound is against v45, which
v52 improves on, so the realised gain will be smaller.

## Addendum F, 25 September 2026: the Munich-only candidate, built and not uploaded

The owner approved the download and the build on 24 September. All 62 scored days were downloaded and extracted
by 02:47 (`fetch_scored_days.py`; no day below 60 % of its month's median point count). `build_munich_candidate.py`
then built the candidate on top of v52: 6,722 scored Munich main-lane rows (25.9 % of Munich's scored departures,
1.95 % of all scored rows) get the detector's taxi time; every other row is byte-identical to v52.

**One post-hoc guard, disclosed.** Inspecting the first build (label-free) showed one row, on 29 July 2026, whose
prediction rose by 3,264 s: the aircraft had been at 148 kt within the 30 minutes before the transition and then
stayed at its stand for 61 minutes. That is docking after a landing more than 10 minutes earlier, which the 600 s
arrival rule does not cover. The guard withdraws an estimate when both hold: faster than 30 kt within 30 minutes
before the transition, and still at the stand more than 20 minutes after it. `stay_guard_check.py` shows it
withdraws **none of the 1,484 estimates on the 13 labelled Munich days**, so no validation result changes. (The plain
stay rule alone would have withdrawn one correct estimate, 6 s off, and was therefore not used.) On the scored days
it withdraws exactly that one row, which keeps v52's 590 s instead of 3,854 s.

**Gates**, all passed: G1 base is v52; G2 detector code unchanged; G3 production test passed (addendum E); G4
submission check without errors, untouched rows identical, all finite; G5 guard neutral on the labelled days.
Candidate: `.cache/munich_candidate/candidate_v52_munich_detector.parquet`, SHA-256 `5f20c51ab0c3bd5f…`
(`results/munich_candidate_manifest.json`). Prediction shift against v52: median -32.6 s, 10th and 90th
percentiles -193 and +143 s.

**Prediction, fixed before any upload.** Gain per covered row against v45: 18,744 (addendum E). v52 is stronger on
these rows by about 705 (sequence nets) and 715 (NM add-on on its covered days), leaving about 17,323 against v52.
At 1.95 % of scored rows that is **264.65 at full transfer, 264.74 at a transfer of 0.85 (central), 264.81 at 0.74**,
against v52's 265.2839. The detector was validated in December and January only; July rows, half the replacements,
are untested in summer, so the outcome could fall outside that range.

## Addendum G, 25 September 2026: the official result

The owner authorised the upload. `research_290/upload_v54_once.py` (the single-shot safeguards of the v52 uploader)
uploaded the candidate once at 07:02 UTC as `upbeat-goblin_v54.parquet` (remote size equal to local, 4,730,688
bytes). **Official score: 264.2328**, against v52's 265.2839: a gain of **1.0511 s**.

That is better than the whole range fixed before the upload (264.65 to 264.81; central 264.74). The realised MSE
reduction, 556.6, is **28,552 per replaced row**: 1.65 times the prediction against v52 and 1.52 times the gain
validated against v45 in January 2025. The most likely reason is July, half the replacements, where the model's
errors are larger (v45 held-out July 231.7 s against January 197.0 s). One board number cannot split it, and no
constant was adjusted from it. Record: `results/upbeat-goblin_v54.json`.
