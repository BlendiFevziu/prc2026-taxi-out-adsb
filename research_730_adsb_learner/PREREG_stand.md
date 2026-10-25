# Pre-registration: stand readout on top of the per-flight ADS-B features (pilot 2)

Written 15 October 2026, **before the stand columns are computed and before any fit**. Same conditions as `PREREG_pilot.md`
(private research, no leaderboard, no upload). Changes go into dated addenda.

## 0. Why

The pilot passed (+13.3 s). The leading team that reports it says the stand readout, the moment the aircraft leaves its
assigned stand, is the most precise form of the ADS-B evidence (median error 39 s against 172 s for the end of the last stop and
128 s for the NM clock). We already hold a stand detector (`research_290/stand_detector.py`, OSM stands, abstains rather
than guesses) built for hand-placed corrections. This pilot feeds its output to the learned model as columns.

## 1. Columns (computed in the same matching pass, `features_adsb.py`, `STAND_MERKMALE`)

For a matched departure with a mapped assigned stand (`STAND_mvt` against the OSM `parking_position` table of its airport),
`detect_onset_v2` is run on the matched segment with the detector's frozen constants (parked within 30 m, stand radius 80 m,
gap at most 120 s): `sd_status` (integer code of the detector's status, 0 when no stand or no tracks), `sd_onset_rel_s`
(take-off minus the midpoint of the bracket, when the detector gives a bracket), `sd_gap_s` (bracket width), `sd_parked_m`
(distance of the parked point from the stand), `sd_run_pts` (points inside the stand radius), `sd_first_dist_m` (distance of the
segment's first point from the stand).

## 2. Arms and protocol

Same rows, folds, seeds, hyperparameters and 1,200 rounds as pilot 1. Arms: **real** (91 + 12 ADS-B, the pilot's cached
out-of-fold predictions), **real2** (real + the six stand columns), **control2_k, k = 1..3** (real + the six stand columns permuted
jointly within airport and day).

## 3. Decision (all must hold)

1. real2 beats the mean of the controls in each of the two seeds;
2. positive in both months;
3. positive after removing the 100 most favourable rows;
4. airport-day block bootstrap 95 % interval above zero;
5. RMSE gain of real2 over the mean control2 at least **0.3 s**;
6. real2 beats real (the pilot's arm).
Reported, not judged: gain by airport, the share of departures with `sd_status` ok, and the gain on rows where the detector gives an
estimate against the rest.

## 4. Meaning

Pass: keep the stand columns; continue with neighbour traffic and the four-month analogue. Fail: the stand readout adds nothing over the
plain track summaries for a learned model, and it is dropped from the rebuild (the hand-placed corrections stay as they were).

## Addendum A, 15 October 2026: pilot 2 passes all six rules

`pilot2_screen.py`, `results/pilot2_screen.json`, log `.cache/research_730/pilot2.log`. The detector gives an estimate on 9.1 % of the 339,046 rows
(30,705; status ok), 8.2 % have the aircraft at the stand but no clean parked-to-motion transition, 50.4 % never enter the 80 m radius, 6.1 % have no
points, 4.7 % a stand the OSM table cannot place or resolve. Out-of-fold RMSE, mean of two seeds: **real2 220.824, mean control2 223.232, real 223.166**. Gain
over the controls **+2.41 s**, over the pilot's real arm **+2.34 s**. Both seeds positive (+1,099 / +1,039 MSE per row), January +1,153, July +1,001, +894
without the best 100 rows, airport-day bootstrap [+927, +1,216]. Per airport against the controls (MSE per row, share of rows with an estimate): LSZH +2.5k (21 %),
EHAM +2.3k (33 %), EDDF +1.8k (5 %), EDDM +1.7k (15 %), EGLL +1.2k (4 %), LEBL +1.0k (13 %), LIRF +0.8k (1 %), LEMD +0.4k (3 %), LFPG -0.2k, LTFM 0. Rows with an
estimate gain +6,048, the rest +573. **Verdict: pass.** The limiting factor is coverage: half the rows never enter the OSM stand radius. Next: stand positions
learned from arriving aircraft (pilot 3), the move the leading team reports for raising coverage.
