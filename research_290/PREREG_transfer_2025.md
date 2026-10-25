# Transfer of the v2 stand detector to other airports: frozen plan

Written 24 September 2026 on branch `claude/munich-stand-detector`, **before any result at these airports was
computed**. Nothing in sections 1 to 5 may change after the first judged run; changes go into a dated addendum.
Private research: no production change, no candidate, no upload, no merge.

## 0. Why

v2 (`PREREG_stand_detector.md`, addenda B and C) passed at Munich against a single LightGBM. Munich is 7.5 % of
the scored departures, so on its own it is worth under 0.9 s on the official scale. Two questions remain open:
does the detector work at other airports, and does it beat production-strength predictions there.

## 1. What is fixed

- **Detector**: `stand_detector.detect_onset_v2` exactly as in ffccea1 (R_STAND 80 m, REST 2 kt, R_PARK 30 m,
  MAX_GAP 120 s, MAX_ONSET_SPREAD 600 s, FAST 30 kt within 600 s). Stand mapping (`lookup`) and matching
  (`offblock_from_tracks.match`, unique claims only) unchanged.
- **Airports**: EHAM, EGLL, LEMD, EDDF, LSZH, LEBL, LFPG. Excluded: LTFM and LIRF (no ADS-B ground points on the
  scored July 2026 samples), and EDDM (decided by the Munich validation, and almost no ground coverage in July 2025).
- **Script**: `research_290/transfer_2025.py`, committed with this document; its verdict logic is unit-tested in
  `tests/test_transfer_judge.py`.

## 2. Part A: December 2025 (the main evidence on transfer)

- **Dates**: the nine December days already downloaded for Munich, in two blocks: {3, 5, 8, 10, 11} and
  {13, 14, 17, 20} December. At these seven airports their labels have never been read.
- **Comparator**: `december_model.py`, one LightGBM on source months 1 to 11. It is weaker than production, so
  gains are overstated.
- A label-free dry run gave estimates per departure: EHAM 14.5 %, EGLL 13.6 %, LEMD 17.3 %, EDDF 6.7 %,
  LSZH 8.8 %, LEBL 17.8 %, LFPG 0.0 % (2 estimates).

## 3. Part B: July 2025 (the production-strength evidence)

- **Dates, not yet downloaded**: two blocks, {2, 9} and {16, 23} July 2025. 13 July is excluded because the old
  detector was evaluated there on 21 September.
- **Comparator**: the full v45 ensemble's held-out main-lane predictions (`ensemble_baseline.parquet`, SHA-256
  checked, 216.77 s on January and July 2025, trained without those months).
- A label-free dry run on 13 July gave: EHAM 17.9 %, LEBL 10.8 %, LEMD 7.0 %, EDDF 3.6 %, LSZH 3.6 %,
  EGLL 1.5 %, LFPG 0 %. Several airports will be "insufficient" in this part. That is expected and is no failure.

## 4. Decision rules, per airport and part

- **V1** MAE against the recorded block at most 60 s and at least 75 % within 60 s, on all estimates.
- **V2** detector RMSE below the comparator's on identical main-lane rows in **each** of the part's two blocks.
  A block with fewer than 20 such rows makes the airport **insufficient** in that part (no verdict).
- **V3** squared-error gain above zero after removing the most favourable 5 % of rows.
- **V4** 2,000-draw flight bootstrap (seed 0) 95 % interval of the gain above zero.
- Verdict per part: **pass** (all four), **fail**, or **insufficient**.

## 5. What the outcomes mean, fixed now

- An airport is **eligible for a candidate build** when it passes Part A and does not fail Part B (pass or
  insufficient). Failing either part excludes it.
- Eligibility is not a submission. A candidate would still need the tracks of all 62 scored days, a full-ensemble
  validation of the substitution on January and July 2025 rows, and the owner's approval before any upload.
- Reported, not judged: the official-scale projection over eligible airports, at validation coverage and at the
  coverage measured on scored 2026 days (`scored_coverage_2026.py`, label-free).

## 6. Honesty notes

- Seven airports in two parts: an airport needs all four rules within a part, which makes a chance pass unlikely,
  though not impossible. Each verdict is reported with its numbers.
- At EHAM, EDDF and LSZH only about a third of departures match an ADS-B visit, against about 90 % elsewhere; the
  callsign mapping is the likely cause. Improving it is a separate change for later and is not part of this test.
- The detector's rules were found at Munich; nothing here was tuned at these airports.

## Addendum A, 24 September 2026: Part A result; Part B not run

Part A run once at 10:15 UTC at commit ad926dd: `transfer_2025.py december`, output
`research_290/results/transfer_december.json`. The label check was exact at every airport. Sections 1 to 5 were not
changed.

| Airport | Estimates | Median error | MAE | Within 60 s | Detector vs comparator RMSE | V1, V2, V3, V4 | Verdict |
|---|---|---|---|---|---|---|---|
| EHAM | 818 | -19.1 s | 63.1 s | 78.9 % | 104.6 vs 147.6 s | fail, pass, pass, pass | fail |
| EDDF | 358 | +20.8 s | 75.1 s | 64.5 % | 125.8 vs 184.3 s | fail, pass, pass, pass | fail |
| LEBL | 703 | +16.1 s | 78.3 s | 70.7 % | 158.4 vs 177.7 s | fail, pass, fail, pass | fail |
| LSZH | 280 | +37.1 s | 138.2 s | 55.0 % | 203.1 vs 172.1 s | all four fail | fail |
| LEMD | 923 | +117.6 s | 192.6 s | 18.0 % | 269.8 vs 167.2 s | all four fail | fail |
| EGLL | 800 | +275.5 s | 314.3 s | 2.6 % | 419.6 vs 229.3 s | all four fail | fail |
| LFPG | 2 | | | | | | insufficient |

**Verdict, as fixed in section 5: no airport is eligible for a candidate build.**

**Part B was not run.** Eligibility requires a pass in Part A, so Part B cannot change any verdict. Downloading
four July 2025 days (about 13 GB, at download speeds that stalled twice that morning) for a descriptive number was
deferred. It is recorded here so that the plan and what was actually run are both on file.

**Post-hoc, description only.** Where the detector fails it is late. Estimates more than 120 s after the recorded
block: EGLL 80 %, LEMD 50 %, LSZH 41 %, EDDF 20 %, LEBL 16 %, EHAM 14 %, against 3.4 % beyond 120 s either way
at Munich (17 of 498, v2 validation). Early misses beyond 120 s are 0 to 2 % everywhere. EHAM is centred on the
label (median -19 s, quartiles -30 and +31 s) and fails V1 only through its late tail (90th percentile +184 s).
The pattern matches the push-back seen only in motion at Munich's stands 181 to 189 (`PREREG_stand_detector.md`,
addendum A): no parked point is observed before the push-back, the aircraft stops after it within 30 m of a stand
vertex, and the detector takes the taxi start. At Munich that class was a minority; at EGLL and LEMD it dominates.
A rule against it would be found after this result and needs its own registration and fresh dates. Where the
detector does beat its comparator (EHAM, EDDF, LEBL), that comparator is a single LightGBM, and the
production-strength Part B was not run.
