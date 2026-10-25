# Pre-registration: stand positions learned from arriving aircraft (pilot 3)

Written 15 October 2026, **before any code for it is run and before any fit**. Same conditions as the earlier pilots (private research,
no leaderboard, no upload). Changes go into dated addenda.

## 0. Why

Pilot 2 gave +2.4 s from the stand readout, but the detector gives an estimate on only 9.1 % of rows: 50.4 % never enter the 80 m radius of the OSM stand,
4.7 % have a stand the OSM table cannot place, and OSM positions need not match where aircraft really stop. jovial-uniform reports learning stand
positions from arriving aircraft (929 stands at five airports, 3 m median scatter) as its most precise stand source. Arriving aircraft end their roll-in
exactly at the assigned stand, and every arrival of the 2025 files carries `STAND_mvt` (100 %) and a callsign (99.3 %).

## 1. Learned stand table (`research_730_adsb_learner/features_stand_learned.py`)

For each arrival (`PHASE_mvt == ARR`, airport = `ADES_mvt`), take the track points of its callsign at that airport from 1 minute before to 45 minutes after landing
(`MVT_TIME_UTC_mvt`); the longest stationary episode (on the ground, below 2 kt, at least 60 s) starting after landing is the parked position, its median
latitude and longitude. Per airport and normalised stand key: keep the stand when at least **3** parked positions exist and **all lie within 30 m of their median**
(otherwise the stand is dropped as unreliable, no tuning); the learned position is the median. Built per airport from all arrivals of the track days
of the scored months (static physical positions, no departure label of any kind; the test months' arrivals and tracks are equally available at prediction time).

## 2. Columns

The six `sd_` columns of pilot 2, computed with `detect_onset_v2` against the learned stand position (one vertex), named `sl_status`, `sl_onset_rel_s`, `sl_gap_s`,
`sl_parked_m`, `sl_run_pts`, `sl_first_dist_m`. A stand missing from the learned table gets `sl_status` 8.

## 3. Arms and protocol

Rows, folds, seeds and hyperparameters as pilots 1 and 2. Arms: **real2** (pilot 2's cached arm: 91 + 12 ADS-B + 6 OSM stand columns), **real3** (real2 + the six `sl_`
columns), **control3_k** (real2 + the `sl_` columns permuted jointly within airport and day, k = 1..3).

## 4. Decision (all must hold)

1. real3 beats the mean control3 in each of the two seeds; 2. positive in both months; 3. positive after removing the 100 most favourable rows;
4. airport-day block bootstrap 95 % interval above zero; 5. RMSE gain over the mean control3 at least **0.3 s**; 6. real3 beats real2.
Reported, not judged: the number of learned stands per airport against the OSM stands matched, the share of rows with `sl_status` ok, the gain by airport, and the gain on rows
where the OSM readout abstains.

## 5. Meaning

Pass: the learned stand table is part of the rebuild, and the same learning is repeated on every month we download. Fail: the OSM readout stays as it is.

## Addendum A, 15 October 2026: not fitted, deferred until more months of tracks exist

`pilot3_screen.py prepare` (no fit) and a coverage count (`.cache/research_730/parked_points_1_7.parquet`). Of 343,999 arrivals in January and July 2025, 12,638 (3.7 %) give a
parked position; 1,091 distinct (airport, stand) pairs are seen. The registered rule (at least 3 positions, all within 30 m) learns **66 stands** and covers **2.6 %** of
departures (EDDF 6 of 176 seen, EDDM 11 of 138, EGLL 4 of 136, EHAM 13 of 194, LEBL 17 of 118, LEMD 4 of 111, LFPG 1 of 23). Relaxed to at least 2 positions: 113 stands, 4.0 %; at
least 1 with no agreement check: 379 stands, 14.0 %. The OSM detector already gives an estimate on 9.1 % of rows, so the registered rule cannot add measurable coverage and a
relaxed one adds positions that were never checked. The fits are **not run**. The idea is deferred until at least six months of tracks are held (the leading team uses twelve and
reports 929 stands at five airports); the rule is then re-counted before any fit, with the thresholds unchanged.
