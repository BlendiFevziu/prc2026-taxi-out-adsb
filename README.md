# Taxi-out time prediction with ADS-B ground tracks

Predicting how long a departing aircraft taxis, from off-block to take-off, at ten European airports
(EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM). The project started as team
**upbeat-goblin**'s entry to the [PRC Data Challenge 2026](https://ansperformance.eu/) of the EUROCONTROL
Performance Review Commission and the OpenSky Network, and continued after the challenge closed on
14 October 2026 with one question: how much do open ADS-B ground tracks add to a strong tabular model?

**Answer: 15 to 30 seconds of RMSE per departure**, on months the model never saw.

## Results

Main-lane RMSE in seconds (departures with a Network Manager off-block clock, about 98 % of all rows),
on held-out 2025 months:

| Held-out months | Ensemble alone | Ensemble + ADS-B corrector | Gain |
|---|---:|---:|---:|
| January / July | 216.8 | 201.0 | **+15.8** |
| February / June | 209.6 | 191.2 | **+18.5** |
| November / December | 194.4 | 164.7 | **+29.7** |
| Trained on one month pair, applied to another | 220.2 | 205.0 | **+15.2** |

During the challenge the best leaderboard score was 245.046 s RMSE (all rows of January and July 2026,
version v80). The leaderboard scores all rows, including the rare departures without an off-block clock,
so it is not on the same scale as the table above.

## How it works

Two stages.

1. **Base ensemble** (`main_model.py`, `research_250/`, `research_260/`): five LightGBM, three CatBoost and
   five neural networks, ten per-airport models and a reporting gate, plus five sequence networks that see
   each flight's neighbours at the same airport. It predicts the gap between the airport's block time and the
   Network Manager's off-block clock. A separate fallback lane handles departures without that clock.
2. **ADS-B corrector** (`research_730_adsb_learner/`): a residual LightGBM over the base's prediction.
   Twelve columns come from the aircraft's own ground track in the public
   [adsb.lol](https://adsb.lol) archive, matched by callsign (first and last ground point, speeds, stops,
   movement start, distance, gaps); six more come from a stand detector built on OpenStreetMap stands.

Every change was decided by a rule written before the run (`PREREG*.md` in each research folder): better in
every seed and every month, still better without the 100 most favourable flights, an airport-day bootstrap
interval above zero, and at least 0.3 s. Ideas that failed are kept with their numbers. The full account of
the ADS-B work, including fifteen tests and what they settled, is
[`research_730_adsb_learner/REPORT.md`](research_730_adsb_learner/REPORT.md).

## Repository layout

| Path | Contents |
|---|---|
| `main_model.py`, `neural_model.py`, `features_*.py`, `submission.py` | the base model, its features and the submission format checks |
| `research_*/` | one folder per question: code, pre-registrations, results (map: [`RESEARCH_INDEX.md`](RESEARCH_INDEX.md)) |
| `research_730_adsb_learner/` | the ADS-B features, the corrector pilots and the 2026 build |
| `adsb-lol/` | download and extraction of the adsb.lol daily archives |
| `geometry/` | OpenStreetMap airport geometry (ODbL) |
| `tests/` | unit tests (`python3 -B -m unittest discover -s tests`) |

## Reproducing

1. Python 3.9 with the pinned packages: `pip install -r requirements.txt`. On macOS import torch before the
   tree libraries (the scripts do).
2. The challenge data is **not included**; it is issued per team. See [`data/README.md`](data/README.md) and
   [`SETUP.md`](SETUP.md).
3. Tracks: `research_730_adsb_learner/fetch_months_loop.sh <YYYY-MM> ...` downloads and extracts the adsb.lol
   days (about 3.3 GB read per day, 12 MB kept).
4. The base ensemble: `research_250/release_pipeline.py` (about 8 hours on a laptop). The corrector and the
   2026 file: `research_730_adsb_learner/build_2026_final.py table`, then `build`.

Trained weights are not in the repository.

## Data and licences

Code: GPLv3 (`LICENSE`). ADS-B tracks: adsb.lol contributors, ODbL 1.0. Airport geometry: OpenStreetMap
contributors, ODbL 1.0. The challenge data belongs to its publishers and is not redistributed here.
