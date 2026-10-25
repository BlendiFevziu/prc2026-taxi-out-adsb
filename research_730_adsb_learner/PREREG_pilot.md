# Pre-registration: pilot of per-flight ADS-B features in a learned model (all of January and July 2025)

Written 15 October 2026, **before any code is run and before any fit**. Private research for the owner's own project
after the competition's close (14 October, 10:00 UTC); no leaderboard, no upload, no organiser contact. Changes go into
dated addenda.

## 0. Why

The teams ahead of us report their largest model-side gain from feeding ADS-B ground-track features of every departure
into a learned model (jovial-uniform -13.7 s on four months of 2025; amazing-marshmallow -14.8 board seconds;
vigorous-whistle's September drop). Our own ADS-B use was a hand-built stand detector and correction layers fitted on
about nine development days, and the line was closed on 21 September after one measured day (13 July 2025, 5.1 % of
departures with usable push-back evidence). This pilot tests the learned-model use on all of January and July 2025,
which we already hold as extracted tracks (`../adsb-lol/tracks/`, 63 days).

## 1. Columns (`research_730_adsb_learner/features_adsb.py`)

Per departure (matched by FLIGHT_mvt/CALLSIGN_flt and airport, track points of that callsign at that airport in the 90
minutes up to 2 minutes after take-off, last contiguous ground segment, gap above 20 minutes splits a segment), all
label-free: `ad_matched`, `ad_pts`, `ad_first_rel_s` (take-off minus first point), `ad_span_s`, `ad_gs_first`,
`ad_gs_max`, `ad_last_stop_rel_s` (take-off minus the last point with speed below 1 m/s), `ad_move_rel_s` (take-off
minus the start of the final move), `ad_dist_m`, `ad_gap_max_s`, `ad_stops` (stop episodes over 30 s),
`ad_airborne_first` (the first point is already airborne).

## 2. Arms and protocol

Rows: the 91-feature release tables of January and July 2025 (`months_1_7.parquet`, main lane `hat_aobt == 1`, label
present), joined to the tracks. Four day-blocked folds (days assigned to folds by a fixed hash of the date, so a day is
never split, both months in every fold); each fold predicted by a model trained on the other three.
Arms: **base** (91), **real** (91 + 12), **control 1 to 3** (the 12 columns permuted jointly within airport and day).
One LightGBM, learning rate 0.03, 255 leaves, min_child_samples 50, subsample and colsample 0.8, 1,200 rounds, target
y minus ersatz_s, prediction max(ersatz_s + model, 300), two training seeds (177, 277).

## 3. Decision (all must hold; this is a pilot, so the bar is the screens' bar)

1. real beats the mean control in **each** of the two seeds;
2. positive in **both** months;
3. positive after removing the 100 most favourable rows;
4. airport-day block bootstrap 95 % interval above zero;
5. RMSE gain over the mean control at least **0.3 s**;
6. real beats the base.
Reported, not judged: gains by airport, coverage of matched rows by airport, and the gain on rows with and without a match.

## 4. What the outcome means

Pass: ADS-B per-flight features are the first part of the rebuild, and the next step is the stand readout and the ADS-B
stacker on top. Fail: the earlier closure stands for this form, and the gap to the leading teams lies elsewhere
(the Rome layer, the daily totals, the stack), to be tested in that order.

## Addendum A, 15 October 2026: pilot passes all six rules

`pilot_screen.py`, `results/pilot_screen.json`, log `.cache/research_730/pilot.log`. 339,046 main-lane departures of January and July 2025, 62 track
days (the registration said 63; one of the 63 files is a probe file, not a day), 79.7 % of departures matched to track points. Out-of-fold
RMSE, mean of two seeds: **real 223.166, mean control 236.494, base 236.442**. Gain over the controls **+13.33 s**, over the base **+13.28 s**.
Every seed positive (+6,127 / +6,126 MSE per row), January +5,551 and July +6,595, +5,702 without the best 100 rows, airport-day bootstrap
[+5,579, +6,745]. Per airport against the base (MSE per row, share of rows matched): LSZH +12.6k (98 %), EHAM +11.9k (99 %), LEBL +10.2k
(99 %), EDDF +10.5k (98 %), EDDM +9.3k (99 %), LIRF +8.0k (69 %), EGLL +3.0k (99 %), LEMD +1.6k (78 %), LFPG +0.2k (89 %), LTFM +0.1k (0 %).
Matched rows +7,702, unmatched rows -180. **Verdict: pass.** Reading: the gain follows the track coverage and comes from the airports
where tracks include the stand area; LFPG, LEMD and LTFM gain little, so the hubs the leading teams score on are our weak spot.
Limits: the base is one 91-feature LightGBM (236.4), not the shipped ensemble, so the gain over a stronger stack will be smaller; folds are
day-blocked within the same two months, so neighbouring days share weather and traffic; the four-month analogue (Feb, Jun) waits for the downloads.
