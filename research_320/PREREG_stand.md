# Pre-registration: clean stand observations as off-block evidence on top of v56

Written 27 September 2026 on branch `claude/stop-move-episodes`, **before any model on these rows was fitted and
before any label of a stand-observed row was read for this question**. Owner's go-ahead: "yes run the stand test".
Open data only (challenge rows, adsb.lol tracks, OSM and X-Plane stands); no NM input of any kind. No upload, no merge.
Reference: scored v56, 261.2616, SHA-256 `1767cc0257ddebf6…`.

## Why

`OPEN_DATA_CEILING.md`: the one untapped open-ADS-B pool with real coverage is departures seen parked at the assigned
stand (`observed_at_assigned_stand` in Codex's frozen classifier: two still points within 30 m of the stand spanning
60 s or more). v56 corrects only `partial_ground` rows, and the stand detector is deployed only at EDDM and EHAM, so at
EDDF, EGLL, LEBL and LSZH these rows keep the comparator's value. In the 2026 scored set they number 13,750 in the
seven deployable cells below (Codex's `full_coverage_2026.json`).

## Rows, cells, dates

- **Rows**: main-lane `observed_at_assigned_stand` departures, rows and features from Codex's own
  `multiairport_screen.extract()` in the frozen checkout of 9884150 (same matching, stand map, categories, 23 summaries).
- **Deployable cells (decided now)**: EDDF January and July, EGLL January, LEBL January and July, LSZH January and July.
  EGLL July is excluded before any label: 2026 sees half its flights at the stand, July 2025 only 1-3 %
  (`WITHHELD_COVERAGE_RESULT.md`); it is reported, never judged or deployed. EHAM (Amsterdam detector) and EDDM (Munich
  detector) are not touched.
- **Training**: v56's training dates (January 8, 9, 24, 30; July 2, 4, 18, 23, 2025), stand rows at all five airports.
- **Development evaluation**: January 4, 13, 14, 20, 21, 26, 28 and July 9, 11, 13, 14, 16, 25, 27, 2025 (v56's
  development-test and replication dates; at least two days from every training date; their stand rows were never
  evaluated by anyone).
- **Confirmation, only after a development pass**: the eight unread reserve dates, January 1, 2, 6, 18 and July 7, 12,
  26, 31, 2025 (`research_300/DATE_RESERVATION.md`).

## Features and arms

36 columns: v56's 35 (six context fields, 23 two-hour summaries, five airport indicators, July) plus
**`detector_taxi_s`**, the validated stand detector's taxi time on the same matched visit and stand: take-off minus the
onset midpoint of `stand_detector.detect_onset_v2` with v54's arrival-and-stay guard and the tow and false-onset guard
(`research_300/tow_guard.py`), missing when any of them abstains. Only points of the matched visit; take-off is the
challenge row's movement time. No target-derived input, no ID, no absolute date.

v56's recipe unchanged: target label minus comparator, LightGBM 200 trees, learning rate 0.03, 7 leaves, min leaf 80,
L2 100, row fraction 0.8, column fraction 0.9, max_bin 63, four threads, seeds 177/277/377, corrections averaged,
shrink 0.5, floor 300.

- **real**: all 36 columns.
- **context**: the 24 track columns (23 summaries and the detector) set to zero.
- **control 4101 to 4105**: the 24 track columns permuted jointly within airport and date, separately in training and
  evaluation rows. Same width as real.

The comparator of every rule is **v56's policy on these rows**, which is the pre-NM comparator itself (v56 leaves them
unchanged; there are no NM rows on 2025 dates).

## Rules (development; the confirmation repeats 1 to 7)

Gain per row = squared error of v56's policy minus squared error of the arm, over the deployable cells.

1. pooled gain above zero;
2. above the context arm and each of the five controls;
3. January and July each above zero;
4. above zero after removing the 100 most favourable rows;
5. airport-day block bootstrap (2,000 draws) 95 % interval above zero;
6. each seed above zero on its own (shrink 0.5, floor 300);
7. at least three of the four airports above zero;
8. **minimum meaningful gain**: the sum over the seven cells of (2026 stand rows in the cell x 56/62, a conservative
   allowance for NM-protected days) x (development gain per row in the cell), divided by 344,841, must reach 0.3 s at
   full transfer.

Reported, not judged: per cell (EGLL July too), per date and seed, without days 6, 13 and 20. **A candidate may only be
built if both the development and the confirmation pass**; then on the exact v56 base with every untouched row,
including every NM row, byte-identical, and never uploaded without the owner. Failing either stage stops the line; the
models stay frozen and nothing is re-tuned on these dates.

## Addendum A, 27 September 2026: both stages pass; candidate v57 built and verified, not uploaded

**Development** (`results/stand_development.json`, 14 dates, 3,072 rows): all eight rules. +6,246 MSE per row (RMSE
179.26 to 160.89 s), +3,777 without the best 100, airport-day [5,151, 7,447], January +6,119, July +6,396, all four
airports and all seven cells positive, seeds +6,245 / +6,247 / +6,233; context +133, controls +36 to +161; planning
+0.42 s at full transfer. **Confirmation** on the eight unread reserve dates (`results/stand_confirm.json`, 1,705 rows):
all seven rules. +6,763 MSE per row (186.10 to 166.95 s), +3,288 without the best 100, airport-day [5,596, 8,085],
January +6,908, July +6,612, all cells positive, seeds +6,765 / +6,793 / +6,716; controls +2 to +199.

**Build** (`build_stand_candidate.py`, `results/stand_candidate.json`), on the exact scored v56 file, copied read-only
from Codex's cache with the owner's permission (SHA-256 `1767cc02…` verified; his 2026 context and feature tables
verified against his recorded hashes). The 2026 re-extraction with the detector column reproduces his 23 features and
categories exactly on all 56 days used. Eligible rows exclude days 6, 13 and 20 entirely (every NM row lies there);
on every eligible row the context comparator equals v56's value. A safety gate added before any result was known,
v56's own drift rule per cell, **removed both LSZH cells** (January: first-point stand distance shifted 1.01 training
SD; July: last-point distance 1.26 SD); nothing was loosened to keep them. Kept: EDDF January and July, EGLL January,
LEBL January and July. **8,676 rows change**; every other row is byte-identical to v56; reverse-order replay equal;
submission check clean. Candidate SHA-256 `88245bfc3c40cce8…`, private, in `.cache/research_320/stand_candidate/`.

**Prediction fixed before any upload**: against v56's 261.2616, the kept rows times the per-cell gain per row give
-0.41 s at full and -0.20 s at half transfer with the confirmation gains (held-out fresh dates), -0.28 s at full with
the development gains: **261.06 to 260.85, central 260.98**. No constant was taken from any leaderboard score. Uploading
is the owner's decision.

## Addendum B, 27 September 2026: the official result

Before the upload, `verify_stand_candidate.py` passed all seven checks the owner set (commit ceb2b77). The owner
authorised one submission; `upload_v57_once.py` uploaded the candidate once at 15:43:32 UTC as
`upbeat-goblin_v57.parquet` (remote size equal to local, 4,728,002 bytes). **Official score 260.9292**, against v56's
261.2616: a gain of **0.3324 s**, inside the range fixed before the upload (261.06 to 260.85). Realised 173.6 MSE per
scored row against 212.7 predicted from the confirmation dates at full transfer (0.82) and 147.9 from development.
Record: `results/upbeat-goblin_v57.json`. Nothing was tuned from this score.
