# Munich stand detector and the NM timing control: report, 24 September 2026

Private research on branch `claude/munich-stand-detector`. No production change, no candidate, no upload, no merge.
Rules and verdicts: `research_290/PREREG_stand_detector.md` (sections 1 to 4 and addenda A to C).

## Summary

1. **Timing control (Codex, private NM data).** The NM add-on's gain is carried by the row-level values of the
   timing fields. Against controls of the identical schema they are worth 7,661 MSE per row, robustly, and time
   of day adds nothing. It is attribution on dates inspected before, not a leaderboard gain, and not a physical
   push-back time.
2. **Detector v1 failed** its pre-registered validation (3, 10, 17 and 20 December 2025). The bulk transferred
   (MAE 47.2 s against the recorded block), but one arrival transition, 3,179 s early, broke V2, V3 and V4.
3. **Detector v2 passed** on four fresh dates (5, 8, 11 and 14 December). It adds one rule, registered and pushed
   before those dates were fetched. On 496 identical main-lane rows the detector scores **59.55 s against the
   comparator's 158.64 s**, better on every date. The new rule caught three more errors of 51 to 54 minutes.
4. **The detector does not transfer.** Pre-registered at seven other airports on nine December days, no airport
   passes. EHAM and EDDF beat the comparator but miss the accuracy bar; EGLL and LEMD run minutes late.
5. **Munich beats production strength.** On four January 2025 days against the held-out v45 full ensemble the
   detector scores 65.6 s against 151.8 s, better on every date; all four pre-registered checks pass.
6. **Official result: v54 = 264.2328, the new best, 1.0511 s better than v52.** The prediction fixed before the
   upload was 264.74 (range 264.65 to 264.81); the detector saved 1.5 times its validated gain per replaced row,
   most likely because July rows carry larger model errors. 240 s is still 24.2 s away.

## 1. Codex's timing control: what it shows and what it does not

Source: `private-nm-archive/full-month-research/timing_control/` (outside git). `PROTOCOL.md` was written before
the fits; `private/results.json` is marked COMPLETE at 23 September 18:47 UTC. Private NM research, no
submission, no production change.

**Question.** Does the NM timing family add information beyond continuous UTC time? It had to be asked because
`nm_archive_elapsed_s` turned out to equal UTC seconds of day, so the family's earlier gain could have been a
disguised clock. All arms share one ordered 15-column schema (seven base features, eight timing fields), the frozen
`only_timing` settings, model seeds 13, 21 and 29, no early stopping. Controls are refitted under five
permutation seeds: *shuffle* permutes all eight timing fields jointly within airport and date, *UTC* applies the
same shuffle and then restores only the UTC seconds.

Primary cohort, 52 dates and 288,207 rows (the supplementary 56-date cohort in brackets):

| Arm | RMSE (s) |
|---|---|
| Ensemble baseline, no add-on | 244.776 (305.686) |
| Real timing fields | **232.765** (296.155) |
| UTC restored, rest shuffled, five seeds | 248.636 to 248.749 (308.742 to 308.867) |
| All eight shuffled, five seeds | 248.479 to 248.563 (308.613 to 308.709) |

MSE gain per row, averaged over the five control models (primary cohort):

| Contrast | All | January | July | Best 100 removed | Airport-day bootstrap 95 % |
|---|---|---|---|---|---|
| real vs UTC | 7,661 | 6,696 | 8,437 | 7,217 | [7,062, 8,307] |
| real vs shuffle | 7,580 | 6,601 | 8,366 | 7,135 | [6,983, 8,224] |
| UTC vs shuffle | -82 | -95 | -71 | -174 | [-132, -31] |

Real vs UTC is positive at all ten airports, from 392 (LTFM) to 20,935 (LFPG), and for each model seed
(7,679, 7,667, 7,680). The protocol's support rule is met in full: `history_supported: true`.

**What this shows.**
1. The add-on's gain is carried by the row-level values of the timing fields. Every control loses it.
2. Time of day carries nothing. Restoring the UTC seconds to shuffled data makes the control slightly *worse*
   than the fully shuffled one, and reliably so: the bootstrap interval lies entirely below zero and 9 of 10
   airports are negative. `nm_archive_elapsed_s` was not a disguised clock carrying the gain.
3. Every control lands **3.70 to 3.97 s above the baseline**. With the real model's capacity and shrink, a residual
   fit without informative timing values adds noise. So the same-schema contrast (7,661) is the right attribution
   measure, and it is larger than what the real model gains over the baseline: 5,736 MSE, or 12.01 s on primary52.
   The seven base features on their own do not carry the add-on.

**What it does not show.** The 56 dates were inspected before, so this is attribution, not fresh validation.
The bootstrap conditions on the five permutations and misses part of the training uncertainty. The timing family
still contains left censoring and message availability, so the result **does not identify a physical pushback
time**. It is not a leaderboard gain and not a candidate: v52 already ships an NM add-on on the covered rows, and
by the protocol a positive UTC contrast only justifies a separate challenge-only test.

**How it connects to the detector below.** The timing control says NM timing values help without saying they are
a pushback time. The stand detector measures the pushback itself, from positions, and needs no NM archive.

## 2. The detector, frozen on one development day

Rules and their history are in `PREREG_stand_detector.md`, committed as 338885a before any validation date was
opened and pushed with the branch. In short: map `STAND_mvt` to OSM `parking_position` vertices, take the final
run of points within 80 m of that stand before take-off, and read off-block as the **first** parked-to-motion
transition (a point below 2 kt within 30 m of the stand, followed by one at 2 kt or more, or outside 80 m). The
estimate is the midpoint of that bracket. The detector abstains when the bracket exceeds 120 s or when the first and
last transitions lie more than 600 s apart, which is a tow or a reposition.

Development day, Munich, 13 December 2025: 359 departures, 308 uniquely matched to ADS-B visits, stands mapped
for 267 (165 exact, 102 through lettered sub-positions), 41 unmapped.

| Detector | Estimates | Median error | MAE | RMSE | Within 60 s | Within 120 s |
|---|---|---|---|---|---|---|
| Old (`offblock_from_tracks.py`) | 149 | +262.4 s | 245.0 s | 271.5 s | 17 | 31 |
| Frozen stand detector | 90 | **+17.3 s** | **34.4 s** | 62.4 s | **81** | 86 |

Error is estimate minus recorded block time. The rule changed three times on this day, all recorded in the
pre-registration, so these development figures are optimistic by construction; only the validation below counts.

**Coverage on the scored dates (label-free).** The frozen detector, unchanged, on scored Munich departures of
6 and 13 July 2026, matched to truncated archives that hold about a fifth of each day's aircraft: 17 estimates
from 80 matched departures (21.3 %) and 9 from 39 (23.1 %), against 90 from 308 (29.2 %) on the development day.
Gate 0 had shown the feeder coverage itself: Munich aircraft with ground points, 7 of 408 on 13 July 2025,
88 of 106 and 47 of 60 in the July 2026 samples.

## 3. Validation of v1 on four new dates: failed

Run once on 24 September at commit 725de04, on 3, 10, 17 and 20 December 2025, against a single LightGBM trained
on source months 1 to 11 (`december_model.py`: December main lane 197.54 s, Munich 199.29 s, weaker than
production by about 7 s on January and July 2025).

447 estimates from 1,626 departures (27.5 %), 446 of them on the main lane.

| Check | Measured | Outcome |
|---|---|---|
| V1 accuracy against the recorded block | median +26.9 s, MAE 47.2 s, 392 of 447 within 60 s | pass |
| V2 beats the comparator on every date | 302.8 vs 161.9 s on 3 December; 73.3 vs 280.4, 89.0 vs 204.5, 53.5 vs 265.7 | **fail** |
| V3 gain survives removing the best 5 % | -9,278 MSE per row | **fail** |
| V4 bootstrap interval above zero | [-33,885, +73,347] | **fail** |

Pooled: detector 167.29 s, comparator 232.17 s, NM taxi `ersatz_s` 545.01 s on the same 446 rows. **By the rules
fixed in advance, v1 does not justify further work.**

**Why (post-hoc, description only).** One row carries 81.0 % of the detector's squared error: on 3 December at
stand 215 a 53-minute turnaround kept the aircraft inside the stand radius from landing to push-back, docking
produced a still-to-moving step, and the detector took it, 3,179 s early. Within 600 s before that step the
aircraft had been at 142 kt. Without the row the detector scores **73.1 s against 232.3 s** on 445 rows. Every
other miss is under 700 s: 16 late misses of 2 to 6 minutes (11 on stands 181 to 189, where the push-back is not
seen from a parked state and the stop after it lies within 30 m of a second stand vertex), one recorded block that
precedes the movement by about eleven minutes, and one case of speed noise at 2 kt.

**A model-disagreement gate does not work.** Of the four rows where detector and comparator disagree by more than
900 s, the detector is right on three: taxi-outs of 52 to 58 minutes that the comparator underestimated by 17 to 41
minutes. The gate would give up 10.47 million s² to save 10.08 million. The detector's value sits where the model
fails worst.

## 4. v2: one arrival rule, four fresh dates: passed

**The rule** (`stand_detector.detect_onset_v2`): v1 unchanged, but an estimate is withdrawn when the aircraft moved
faster than 30 kt within 600 s before the chosen transition, because that transition is docking after the arrival.
The thresholds are those of the post-hoc classification and were not varied. On the five used dates the rule
removes exactly the failed estimate and nothing else (`v2_used_dates_check.py`). It was registered in addendum B
and pushed (ffccea1) before any data of the fresh dates was downloaded.

498 estimates from 1,688 departures (29.5 %), 496 of them on the main lane.

| Check | Measured | Outcome |
|---|---|---|
| V1 accuracy | median +23.8 s, MAE 36.7 s, RMSE 61.3 s, 443 of 498 (89.0 %) within 60 s, worst -334 and +469 s | pass |
| V2 every date | 41.2 vs 206.1 s (5 Dec), 64.3 vs 117.1 (8 Dec), 67.4 vs 146.6 (11 Dec), 62.0 vs 149.7 (14 Dec) | pass |
| V3 without the best 5 % | +11,666 MSE per row | pass |
| V4 bootstrap | [+17,080, +26,740] | pass |

Pooled: detector 59.55 s, comparator 158.64 s, `ersatz_s` 447.21 s, gain 21,620 MSE per row. A half blend of
detector and comparator scores 87.30 s, worse than the detector alone.

The rule withdrew three estimates. Under v1 they would have been -3,225, -3,090 and -3,259 s, and with them v1 would
have failed V2 on 8, 11 and 14 December. The arrival failure is systematic, about one a day, and the rule removed
no estimate that v1 had right.

## 5. What this means for the competition

**Munich alone is worth less than a second.** The validation script's official-scale upper bound is 0.89 s. It
assumes the single-LightGBM comparator, December coverage and no transfer loss. All three assumptions are
optimistic. Production is about 7 s stronger than the comparator on January and July 2025. The scored July 2026
samples give 21 to 23 % estimates per matched departure against 33 % on the v2 dates. January 2026 coverage has not
been checked.

**The 21 September closure no longer holds as written.** It measured the old detector, which at Munich ran a median
+262 s late on the development day. The new detector sits at +24 s against the same kind of label, and on covered
rows its RMSE is 59.55 s against the comparator's 158.64 s. The other airports have not been re-measured.

**Other airports looked like the lever, and the transfer test (section 6) says they are not, as the detector
stands.** Aircraft with ADS-B ground points on the scored July 2026 samples:

| Airport | 6 July 2026 | 13 July 2026 | 13 July 2025 (labelled) |
|---|---|---|---|
| EHAM | 86 % | 94 % | 87 % |
| EGLL | 87 % | 90 % | 29 % |
| LEMD | 90 % | 90 % | 53 % |
| LSZH | 80 % | 81 % | 76 % |
| EDDF | 70 % | 81 % | 73 % |
| EDDM | 83 % | 78 % | 2 % |
| LEBL | 51 % | 62 % | 53 % |
| LFPG | 49 % | 60 % | 37 % |
| LIRF, LTFM | 0 % | 0 % | 85 %, 0 % |

Share of aircraft seen at the airport that have at least one ground point; the 2026 files are truncated samples.

## 6. The transfer test: the detector does not carry over

Pre-registered in `research_290/PREREG_transfer_2025.md` (ad926dd) before any result at these airports. Part A:
the frozen v2 detector on the nine December days already downloaded, at seven airports whose December labels had
never been read, against the single-LightGBM comparator, with V2 judged on two blocks of dates.

| Airport | Estimates | Median error | MAE | Within 60 s | Detector vs comparator | Verdict |
|---|---|---|---|---|---|---|
| EHAM | 818 | -19.1 s | 63.1 s | 78.9 % | 104.6 vs 147.6 s | fail (V1 only) |
| EDDF | 358 | +20.8 s | 75.1 s | 64.5 % | 125.8 vs 184.3 s | fail (V1 only) |
| LEBL | 703 | +16.1 s | 78.3 s | 70.7 % | 158.4 vs 177.7 s | fail (V1, V3) |
| LSZH | 280 | +37.1 s | 138.2 s | 55.0 % | 203.1 vs 172.1 s | fail |
| LEMD | 923 | +117.6 s | 192.6 s | 18.0 % | 269.8 vs 167.2 s | fail |
| EGLL | 800 | +275.5 s | 314.3 s | 2.6 % | 419.6 vs 229.3 s | fail |
| LFPG | 2 | | | | | insufficient |

No airport is eligible for a candidate. Part B (July 2025 against the v45 ensemble) was not run, because it
could not change any verdict. Post-hoc, the failures are late: the push-back is seen only in motion, a minority
class at Munich (3.4 % of estimates beyond 120 s either way) and the dominant one at EGLL (80 % beyond +120 s)
and LEMD (50 %). A rule against it would be found after this result and would need fresh dates.

## 7. Production strength at Munich: passed

Pre-registered in addendum D (ec63c23): January 2025 is the only month in which Munich had ADS-B ground coverage
(92 % of aircraft) and production-strength held-out predictions exist, those of the v45 full ensemble trained
without January and July 2025.

| Check | January 2025 (8, 16, 24, 26) | Outcome |
|---|---|---|
| V1 accuracy | 454 estimates, median +23.2 s, MAE 36.6 s, 405 (89.2 %) within 60 s | pass |
| V2 every date | 73.6 vs 215.8, 45.1 vs 127.0, 72.7 vs 115.1, 64.4 vs 101.6 s | pass |
| V3 without the best 5 % | +6,527 MSE per row | pass |
| V4 bootstrap | [11,834, 27,080] | pass |

Pooled: detector 65.6 s, v45 ensemble 151.8 s, `ersatz_s` 372.6 s. The gain per row, 18,744, is close to the
21,620 measured against one LightGBM in December, so the weaker December comparator overstated little at Munich.

Reach on the scored days (label-free): 28.1 % of scored Munich departures on 14 January 2026 and 25.9 % on
15 July 2026. Official-scale upper bound against v45: **0.80 s**; v52 is stronger, so about half a second.

## 8. The candidate, built and not uploaded

With the owner's approval all 62 scored days were downloaded and extracted (about 214 GB, finished 02:47 on
25 September), and `research_290/build_munich_candidate.py` built the candidate on v52: **6,722 rows replaced**
(1.95 % of scored rows), all others byte-identical, five gates passed. A disclosed post-hoc guard (addendum F)
withdrew one arrival transition that the 600 s rule missed; it withdraws none of the 1,484 labelled estimates.
File `.cache/munich_candidate/candidate_v52_munich_detector.parquet`, SHA-256 `5f20c51ab0c3bd5f`.

**Prediction fixed before any upload: 264.74** (264.65 at full transfer, 264.81 at a transfer of 0.74), about 0.54 s
better than v52. The detector is untested in summer, and July is half the replaced rows.

**Official result (25 September, uploaded once with the owner's authorisation as v54): 264.2328**, 1.0511 s better
than v52 and better than the whole predicted range. It realised 28,552 MSE per replaced row, 1.52 times the value
validated in January 2025; July rows, where the model errs more, are the likely reason (addendum G).

## 9. Next steps

1. **Done: v54 is the new best.** No constant is to be adjusted from its score.
2. **A v3 for late push-backs**, only with its own registration and fresh dates. EHAM and EDDF would add about half
   a second at most.
3. Corrected 25 September: the NM archives already exist (248 daily members); Codex owns that work, and NM work stays research-only, never in a submission.

## 10. Integrity notes

- **Two looks at December.** v2 is the second attempt on Munich December data, and its rule was found on the first
  attempt's validation dates. The fresh dates guard against overfitting the rule, but a reader should weigh the
  pass as one of two tries. Addendum B recorded before the result that a failure would close the Munich December
  line.
- **The comparator is weaker than production**, so every gain here is overstated for production.
- **Tooling fixes outside git.** `adsb-lol/fetch_days.py` (outside git, no history) had six faults, all found on 23 and 24 September and all fixed:
  a transfer that died halfway left a part the next run skipped as complete (now `.partial`, renamed on success);
  any failure of the second part passed for a short day (now only HTTP 404 does, read with `-w` because this Mac's
  curl reports a 404 as exit 56); a stalled transfer hung for 28 minutes (now aborted below 50 KB/s for 120 s and
  resumed with `-C -`, up to five times); a bare part left by an old run counted as downloaded (now each part needs an
  `.ok` marker and each day a `COMPLETE` marker, without which extraction refuses the day); and only two parts were
  known, while summer 2026 days have three (now parts are fetched until the first 404). A first part alone matters
  because it ends on a tar block boundary and can be extracted quietly as half a day. Every track file used here was
  extracted from a complete download.

## 11. Files

| What | Where |
|---|---|
| Detector (v1 `detect_onset`, v2 `detect_onset_v2`) | `research_290/stand_detector.py`, tests in `tests/test_stand_detector.py` |
| Pre-registration and addenda A to C | `research_290/PREREG_stand_detector.md` |
| Development day | `research_290/munich_dev.py`, `results/munich_dev.json` |
| Comparator | `research_290/december_model.py`, `results/december_model.json` |
| v1 validation and diagnosis | `munich_validate.py`, `munich_validate_posthoc.py`, `results/munich_validate*.json` |
| v2 validation | `munich_validate_v2.py`, `v2_used_dates_check.py`, `v2_withdrawn_rows.py`, `results/munich_validate_v2.json` |
| Transfer test | `PREREG_transfer_2025.md`, `transfer_2025.py`, `results/transfer_december.json`, tests in `tests/test_transfer_judge.py` |
| Reach on scored days (label-free) | `scored_coverage_2026.py`, `results/scored_coverage_2026.json` |
| Production-strength test | `munich_validate_jan2025.py`, `results/munich_validate_jan2025.json` |
| Candidate | `fetch_scored_days.py`, `build_munich_candidate.py`, `stay_guard_check.py`, `results/munich_candidate_manifest.json`, `results/stay_guard_check.json` |
| Coverage | `adsb_coverage_gate.py`, `july2026_detector_coverage.py` and their results |
| Row-level outputs (never committed) | `.cache/munich_stand_detector/` |
| ADS-B tracks (outside git) | `12_Wettbewerb/adsb-lol/tracks/` |
