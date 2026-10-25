# Matching test: the filed callsign as an extra candidate spelling, detector and geometry fixed

Written 25 September 2026 on branch `claude/stand-geometry`, after the geometry test's verdict and **before any label
on the dates below was read** at EHAM or EDDF. Nothing in sections 1 to 4 may change after the first judged run.
Private research: no production change, no candidate, no upload, no merge.

## 0. Why, from label-free evidence only

The failure taxonomy (`research_300/failure_taxonomy.py`, 62 scored days, detector and OSM geometry fixed) loses
**64.9 % of EHAM's and 67.3 % of EDDF's scored departures at matching**: no candidate spelling of `FLIGHT_mvt`
appears anywhere in the day's ADS-B feed. The cause is visible in the data: at these airports `FLIGHT_mvt` carries
the commercial flight number (KL0981, LH002, DE2290) while the aircraft broadcast their ATC callsign (KLM47M,
DLH1TN, CFG290), which no airline-prefix mapping can reconstruct. The filed callsign `CALLSIGN_flt` appears in the
feed for 97.1 % (EHAM) and 94.8 % (EDDF) of those departures on 1 January 2026. At EDDM both fields already carry
the ATC callsign, which is why EDDM is not part of this test.

`CALLSIGN_flt` is a column of the official challenge dataset, not of any third-party NM archive.

## 1. The two arms

- **A (validated)**: candidate spellings of `FLIGHT_mvt` (`offblock_from_tracks.candidates`).
- **B (candidate)**: the same set plus `CALLSIGN_flt` (upper-cased, stripped) when present.
- **Fixed in both**: matching window (visit end within 300 s of the movement time, closest visit), the rule that a
  visit claimed by two departures is dropped, detector v2 plus v54's guard, and the geometry per airport decided by
  the geometry test (addendum B, 8ed8741): **X-Plane-first at EHAM, OSM at EDDF**.

## 2. Dates, rows and comparator

Reserved in `research_300/DATE_RESERVATION.md`: **8, 16, 24 and 26 January 2025; 4, 11, 18 and 25 July 2025**,
at EHAM and EDDF. Main-lane departures; comparator the v45 full ensemble's held-out predictions (SHA-256 checked).
Policy and incremental value exactly as in the geometry test (`research_300/incremental.py`).

## 3. Decision, per airport

The four rules of the geometry test, unchanged: pooled gain per departure above zero; above zero in each month with
at least 20 changed rows; gain per changed row above zero without the most favourable 5 %; airport-day bootstrap
95 % interval above zero. Fewer than 20 changed rows: insufficient. Also reported: coverage per arm, accuracy of the
added estimates against the recorded block, January and July separately, and the official-scale upper bound.

**Descriptive, not judged (declared now):** the same comparison with every detector taxi time above 90 minutes
withdrawn in both arms (`TAXI_CAP_S`). Reason, post-hoc on the geometry dates after their verdict: EHAM's detector
lost to production on its covered rows because of ONE row, a tow from the recorded stand two hours before the
departure (16 July 2025, 7,629 s against a true 363 s); without it the detector scores RMSE 47.0 s there. More
matches can bring more such rows. A tow guard is a new detection rule and gets its own test on fresh dates.

## 4. What the outcome means

A pass makes the filed-callsign rule the matching of any further detector work at that airport. It is not a
submission. The late push-backs that failed the transfer test are a detection problem and are not addressed here.

## Addendum A, 25 September 2026: the result

Run once at 16:16 UTC (`arm_test.py matching`, `results/arm_test_matching.json`) after a label-free dry run; the
code was that of 0c97578 (the only later commit, 3022b87, added the weather pre-registration). Label check exact.

| | EHAM | EDDF |
|---|---|---|
| **Verdict** | **fail** (all four rules) | **fail** (all four rules) |
| Main-lane departures / rows changed | 5,311 / 565 | 4,776 / 232 |
| Coverage, FLIGHT_mvt to plus CALLSIGN_flt | 10.1 to 20.8 % | 3.1 to 7.9 % |
| MAE against the recorded block, both arms | 43.9 and 41.8 s | 105.9 and 104.9 s |
| Policy RMSE on its covered rows vs the comparator's there, plus CALLSIGN_flt | 132.5 vs 137.6 s | 171.4 vs 136.3 s |
| Gain per departure, airport-day 95 % | -243 [-4,087, +1,755] | -533 [-906, -179] |
| January / July, rows changed and gain per departure | 184, -2,926 / 381, +1,934 | 69, -877 / 163, -275 |

**Decision, as fixed in section 3: the filed-callsign rule is not adopted at either airport.** It does what it was
meant to do, doubling EHAM's matched and estimated departures at unchanged accuracy (1,011 of 1,103 within 60 s), but
the added estimates do not beat production in squared error: at EDDF they are poor, at EHAM a few large errors in
January outweigh them. The 90-minute taxi cap (descriptive) withdrew nothing, so those errors are not tows of the
16 July kind.
