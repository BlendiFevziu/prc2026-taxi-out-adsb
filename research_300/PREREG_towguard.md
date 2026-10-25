# Pre-registration: a tow and false-onset guard for the stand detector at Amsterdam

Written 25 September 2026 on branch `claude/stand-geometry`, **before any of the dates below was downloaded and
before any label on them was read**. Private research: no upload, no merge. No third-party NM data is involved.

## Why

The geometry test passed at EHAM (`PREREG_geometry.md`, addendum B), but on its own dates the X-Plane-first
detector reached a policy RMSE of 331.3 s on the rows it covered against 151.8 s for production, because of one row:
UA021 on 16 July 2025 at G73, detector taxi 7,629 s against a recorded 363 s. Without that row the detector's RMSE
there was 47.0 s (`results/posthoc_eham_tail.json`). The matching test's January losses at EHAM came from rows of the
same kind. Nothing about Amsterdam can be decided while one such row can cost more than a day of gains.

Development used **only dates already spent** (the 16 geometry and matching dates): `research_300/tow_guard_dev.py`,
summary in `results/tow_guard_dev.json`. The tracks of the five worst EHAM rows show two mechanisms:

- **False onset.** A single speed reading of 2 to 4 kt at the stand counts as the push-back, and the aircraft then
  rests at its parked spot for many more minutes (QY1239, 16 January: 46 minutes; TK1954, HV6005, HV5679 similar,
  with two or three onsets spread 438 to 565 s, just under the detector's 600 s limit).
- **Tow or reposition.** The aircraft leaves the stand, stops somewhere else for most of an hour, and the airport's
  block time belongs to that later position (UA021: 51 minutes at another stand; KL1713, 8 January: 43 minutes).

## The guard (fixed, `research_300/tow_guard.py`)

Applied only to estimates that `stand_detector.detect_onset_v2` and v54's arrival-and-stay guard already accept.
Only points after the onset and up to take-off are used. The estimate is withdrawn when either holds:

| rule | condition | constant |
|---|---|---|
| `false_onset` | after the onset, points below 2 kt within **20 m** of the aircraft's own parked point (the point at `t_before`) span **300 s** or more | `PARKED_SPOT_M` 20, `REST_AT_SPOT_S` 300 |
| `long_stop` | after the aircraft first leaves the 80 m stand radius, ground points within **30 m** of a stop's first point span **1,800 s** or more before take-off | `STOP_RADIUS_M` 30, `LONG_STOP_S` 1,800 |

On the development dates (3,468 covered departures at EDDM, EHAM and EDDF) the guard withdraws 10: six EHAM rows with
errors of 479 to 7,266 s, three EDDF rows at -119 to -198 s, and one good row (KL0641, +33 s, a 31-minute hold).
It misses EDDF LH492 (+2,066 s), which is a different mechanism. The constants were read off those dates. That is
exactly why they are tested on new ones.

## Dates

Drawn with `numpy.random.default_rng(20260925)`, one `permutation` over the eligible January 2025 days and then one
over the eligible July 2025 days, taking the first four of each. Eligible means never used by any test, dry run or
download attempt (January without 8, 9, 13, 15, 16, 21, 24, 26, 28, 30; July without 2, 4, 9, 11, 13, 16, 18, 23, 25):

- **January 2025: 20, 25, 4, 14**
- **July 2025: 22, 3, 27, 14**

A date that cannot be fetched completely from its preferred release (`research_300/fetch_preferred.py`) is replaced
by the next date of the same month's permutation (January 19, 3, 10, 31, ...; July 5, 10, 8, 12, ...), recorded in an
addendum before any label is read.

## Fixed components

Detector, thresholds and v54's guard unchanged. Geometry at EHAM: X-Plane-first (geometry test). Matching: the
validated `FLIGHT_mvt` spellings (the filed-callsign rule failed, `PREREG_matching.md`). Comparator: the full v45
ensemble's held-out main-lane predictions (SHA-256 checked, `arm_test.comparator`). Main lane only.

## Two comparisons, reported separately

1. **The guard on its own** (A = detector without the guard, B = with it), at EHAM. The four rules of
   `arm_test.evaluate`: pooled gain above zero, each month with at least 20 changed rows positive, gain per changed
   row without the top 5 % positive, airport-day 95 % interval above zero; fewer than 20 changed rows is
   "insufficient". **Expected to be insufficient**: development withdrew 7 of 2,164 EHAM rows (0.3 %), about 1.6
   on 500 fresh rows. Every withdrawn row is then listed with its error, the mechanism named.
2. **Amsterdam viability** (A = production only, B = the guarded detector where it estimates), at EHAM, the same
   four rules. **This is the decision.**

Descriptive, no verdict: the unguarded detector against production at EHAM, and the guard's withdrawals at EDDM
(X-Plane-first) and EDDF (OSM) on the same dates.

## Decision

- **Comparison 2 passes**: the guarded detector is validated at EHAM. An Amsterdam candidate becomes possible, but is
  neither built nor uploaded without the owner, who first has to decide on v52's NM add-on under v54 (the Munich
  X-Plane candidate waits for the same answer).
- **Comparison 2 fails**: the Amsterdam detector stays research-only. The guard is not re-tuned on these dates.
- Either way these dates are spent for Amsterdam.

Run from the repository root, after the downloads: `python3 -u -B research_300/tow_guard_test.py` (`--dry` reports
coverage only and reads no label).

## Addendum A, 25 September 2026: the result

**Data.** All eight registered dates were fetched from their preferred release (`research_300/fetch_preferred.py`,
18:12 to 19:35 UTC): every part's size equals the release listing, every archive ends in the tar end marker, no
replacement was needed. 22 July came from staging-0, the other seven from prod-0. Records in
`results/fetch_preferred_{date}.json`. Test run 19:35 UTC at 14787f9 under the shared lock, label check 0.0 s,
`results/tow_guard_test.json`.

**Comparison 2, the decision: the guarded detector against production at EHAM. Pass, all four rules.**
509 of 5,141 main-lane departures replaced (9.9 %), +1,687.6 MSE per departure; January +1,455.1 (176 rows),
July +1,878.1 (333 rows); +8,758 per replaced row without the top 5 %; airport-day 95 % [1,121.7, 2,239.2].
On the replaced rows the detector's error against the recorded block has median -27.2 s, MAE 40.1 s, 474 of 509
within 60 s; policy RMSE 75.7 s against 150.9 s for production. Naive official scale +0.348 s.

**Comparison 1, the guard on its own: insufficient**, as expected: at EHAM it withdrew one estimate. That one was
**wrong**: AA221 on 22 July at G04, a correct estimate (+23.7 s) of a 62-minute taxi with a genuine 35-minute hold,
where production erred by about 1,800 s; withdrawing it cost 3.32 million squared seconds (-645.6 MSE per
departure). No tow and no false onset occurred among the EHAM rows on these dates: the largest unguarded error was
+646 s.

**Descriptive.** Unguarded detector against production at EHAM: +2,333.2 MSE per departure (+0.482 s), also passing
all four rules, and better than the guarded one on these dates by exactly that one row. EDDM: 666 estimates, none
withdrawn. EDDF: 124 estimates, one withdrawn correctly (LH780, +331 s); one kept error of +3,348 s (CA932, B48), the
EDDF mechanism the guard was never meant to catch.

**What it means.** The Amsterdam detector is validated against production strength on eight fresh dates, as
registered, in its guarded form. The guard's worth rests on the development dates alone, where it caught two tows
worth 59 million squared seconds; across all 24 dates its `long_stop` rule fired four times at EHAM, two tows and two
genuine long holds (KL0641, AA221), while `false_onset` has not yet fired wrongly. Whether to keep, drop or change
`long_stop` is a new question: it needs its own registration and dates, and is **not** decided from this result.
No candidate is built and nothing is uploaded: an Amsterdam candidate waits, like the Munich X-Plane one, for the
owner's decision on v52's NM add-on under v54.
