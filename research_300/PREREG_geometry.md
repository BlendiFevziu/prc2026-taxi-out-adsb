# Geometry test: X-Plane Gateway stands against OpenStreetMap, detector and matching fixed

Written 25 September 2026 on branch `claude/stand-geometry`, **before any label on the dates below was read** at
these airports. Nothing in sections 1 to 4 may change after the first judged run; changes go into a dated addendum.
Private research: no production change, no candidate, no upload, no merge.

## 0. Why, from label-free evidence only

`research_300/geometry_compare.py` on the 62 scored days (no labels exist there), `results/geometry_compare.json`:

| | EDDM | EHAM | EDDF |
|---|---|---|---|
| Scored departures whose stand name maps: OSM / X-Plane / either | 87.0 / 99.9 / 99.9 % | 95.6 / 96.2 / 99.4 % | 96.0 / 86.2 / 97.4 % |
| Flights observed parked for at least 600 s | 3,126 | 915 | 38 |
| Flight-weighted median distance from where they parked to the map: OSM / X-Plane | 16.0 / 5.8 m | 6.6 / 6.0 m | 16.5 / 5.4 m (18 flights) |
| Share within the detector's 30 m parked radius: OSM / X-Plane | 91.0 / 98.6 % | 99.1 / 96.2 % | 88.9 / 88.9 % |
| Push-back bearing against X-Plane heading + 180 degrees | median 1.3, 94 % within 30 | 2.3, 84 % | 11.7, 61 % (18) |

The approved Gateway sceneries: EDDM 111124 (approved 5 June 2026), EHAM 111943 (31 August 2026), EDDF 103368
(16 August 2024); parsed rows 1300/1301 in `research_300/geometry/xplane_{ICAO}.json`, raw zips in `.cache`.

The expected effect is largest at EDDM (13 points more of stand names mapped, and parked positions three times
closer), small at EHAM, and near zero at EDDF, where aircraft are almost never observed parked: **a better map
cannot compensate for an aircraft never observed parked**, and the failure taxonomy measures how much that is.

## 1. The two arms

- **A (validated)**: OSM stands, `stand_detector.stand_table` and `lookup`.
- **B (candidate)**: for each stand name, the X-Plane Gateway position when the X-Plane table maps it under the same
  `lookup` rule, otherwise the OSM stand. Chosen from the table above before any label was read.
- **Fixed in both**: detector `stand_detector.detect_onset_v2` (ffccea1) plus v54's disclosed arrival-and-stay guard
  (`arm_test.guarded`, same constants); matching on the `FLIGHT_mvt` spellings (`offblock_from_tracks.candidates`,
  reproduced exactly by `arm_test.match_candidates`, unit-tested).

## 2. Dates, rows and comparator

- **Dates** (reserved in `research_300/DATE_RESERVATION.md`, de32596): 9, 13, 21 and 30 January 2025; 2, 9, 16 and
  23 July 2025. Airports EDDM, EHAM, EDDF. Munich's July rows are expected to be almost empty (no ground coverage in
  July 2025); that is reported, not replaced.
- **Rows**: main-lane departures (AOBT_3 present) on those dates. **Comparator**: the full v45 ensemble's held-out
  predictions (`ensemble_baseline.parquet`, SHA-256 checked, trained without January and July 2025).
- **Policy per arm**: the arm's detector taxi time where it gives an estimate, the comparator's prediction elsewhere.
  Incremental value of B = per departure, A's squared error minus B's (`research_300/incremental.py`, unit-tested).

## 3. Decision, per airport

B is **adopted at an airport** when all four hold there:
1. the incremental gain per departure, pooled over its dates, is above zero;
2. it is above zero in each month that has at least 20 changed rows;
3. the mean gain per changed row stays above zero after removing the most favourable 5 % of changed rows;
4. the 2,000-draw airport-day block bootstrap 95 % interval of the gain per departure lies above zero.

Fewer than 20 changed rows in total gives **insufficient** (no decision; A stays). Reported, not judged: coverage per
arm, accuracy against the recorded block per arm, each policy's RMSE against the comparator's on its covered rows,
January and July separately, the pooled result over the three airports, and an official-scale figure per airport
(share of scored departures times the gain per departure), which is an upper bound.

## 4. What the outcome means

An adopted geometry becomes the fixed geometry of the next test (matching, `PREREG_matching.md`) at that airport.
Adoption is not a submission: any candidate would need its own build, a full-ensemble check and the owner's decision.

## Addendum A, 25 September 2026: one reserved date does not exist at the source

Written before the first judged run and before any label on the dates was read. **21 January 2025 has no adsb.lol
release**: its first part returns HTTP 404 (checked twice by the downloader and once by hand). It is **replaced by
28 January 2025** (the next Tuesday, used nowhere before; its release exists), fixed here before it is downloaded.
Nothing else changes. `arm_test.TESTS["geometry"]["dates"]` is updated in the same commit.

## Addendum B, 25 September 2026: the result

Run once at 16:09 UTC at commit 7d8b6cd (`arm_test.py geometry`, `results/arm_test_geometry.json`), after a
label-free dry run. Label check exact. Sections 1 to 4 unchanged.

| | EDDM | EHAM | EDDF |
|---|---|---|---|
| **Verdict** | **pass** | **pass** | **fail** (airport-day interval) |
| Main-lane departures / rows changed | 3,478 / 201 | 5,212 / 297 | 4,669 / 55 |
| Coverage, OSM to X-Plane-first | 13.8 to 16.0 % | 13.4 to 9.4 % | 3.4 to 2.7 % |
| MAE against the recorded block, OSM to X-Plane-first | 37.3 to 30.3 s | 84.5 to 49.0 s | 100.3 to 71.1 s |
| Gain per departure (MSE), airport-day 95 % | +958 [133, 2,234] | +1,183 [320, 2,556] | +271 [-37, 505] |
| Gain per changed row without the best 5 % | +4,578 | +6,220 | +13,739 |
| January / July, rows changed and gain per departure | 181, +2,119 / 20, +142 | 141, +924 / 156, +1,403 | 23, +413 / 32, +169 |
| Policy RMSE on its covered rows, against the comparator's there (X-Plane-first) | **44.0 vs 133.2 s** | **331.3 vs 151.8 s** | 141.1 vs 192.0 s |

**Decision, as fixed in section 3**: X-Plane-first geometry is adopted at EDDM and EHAM; OSM stays at EDDF.

**What the pass does and does not say.** At EDDM the geometry adds estimates, makes them more accurate, and the
detector's policy is far better than production on its rows. At EHAM the gain comes mostly from estimates that
X-Plane-first no longer makes: the detector itself is still **far worse than production** on the rows it covers
(331.3 against 151.8 s), despite a median error of -26 s, so a few very large errors dominate. The geometry verdict is
a comparison of two maps; it does not make the detector usable at EHAM. Pooled over the three airports (descriptive)
the incremental gain is +806 MSE per departure, airport-day 95 % [338, 1,440].

## Addendum C, 25 September 2026: correction to addendum A

Addendum A said 21 January 2025 "has no adsb.lol release". **That was wrong.** Its preferred release
(`PREFERRED_RELEASES.txt` of adsblol/globe_history_2025) is a single unsplit prod-0 file,
`v2025.01.21-planes-readsb-prod-0.tar` (1,960,816,640 bytes, under the 2 GB split size), and `fetch_days.py` only
knew the split names `.tar.aa`/`.tar.ab`, so it got HTTP 404 and gave up. The replacement by 28 January stands: it was
fixed before any label was read and the judged run used it. 21 January was fetched afterwards through the preferred
release and evaluated **report-only, with no decision** (`research_300/REPORT_2025-01-21.md`).
