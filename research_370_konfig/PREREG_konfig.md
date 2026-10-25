# Pre-registration: runway configuration and destination bearing as main-lane columns

Written 29 September 2026 on branch `claude/stop-move-episodes`, **before the coordinate file is downloaded and
before any fit**. Nothing below changes after the first fit; changes go into dated addenda. NM-free: only the
challenge's own columns and OurAirports (public domain). Private research: no production change, no candidate, no
upload, no merge.

## 0. Why, and what is already known

- **Where the error is.** 65.2 % of the remaining main-lane squared error sits at airports without track extraction
  (`research_360_integration/PART1_RESULT.md`). The largest 1 % of main-lane rows carry 34.6 % of it, and LIRF,
  LTFM and LFPG hold 26.6, 15.7 and 15.5 % of that top 1 % (`research_320/leads_diagnostic.py`).
- **Origin.** A proposal of 29 September offered four columns for these hubs. Two were dropped in review: OSM
  topology per airport (one constant per airport carries nothing beyond `ADEP_mvt`, since all ten airports are in
  training and test; the per-row OSM signal already exists as `strecke_m`) and a DDR2 congestion index (EUROCONTROL
  NM data, against the owner's NM-free rule of 28 September and the competition's open-data condition). The other two
  are tested here in adapted form. The proposed departure-runway signature is replaced by an arrival-derived one,
  because EDDF 18 is a departures-only runway (`betriebsrichtung.py`); it enters as a plain categorical, without the
  proposed target encoding, which would leak unless fitted across days. The proposed destination quadrants are
  replaced by sine and cosine of the bearing, because a fixed N/S/E/W split cuts arbitrarily at each airport.
- **Already in the 91 features:** the departure runway of every row (`RUNWAY_mvt`, 100 % filled in the ranking
  file), the destination (`ADES_mvt`, categorical), the runway's share of the airport's departures within an hour
  (`qt_bahn_anteil_60`) and the runway queue counters (`research_250/features_correct_queue.py`).
- **Earlier evidence.** Correcting v45's main-lane residuals by airport and operating direction removed 144.2 total
  MSE in-sample and 4.7 cross-fitted by day parity (`research_240/main_weather_diagnostic.py`, 13 September). That was
  a bin mean outside the model, and its direction was the modal runway number of `betriebsrichtung.py`.
- **A correction found while designing (label-free).** Counted by flow side (section 1), every hour at all ten
  airports is 99 to 100 % one-sided in 2025, with 0.29 (EGLL) to 2.17 (LSZH) reversals per day. The 10 September
  record that LFPG (66 %) and LTFM (68 %) "really run mixed" (experiments.csv) counted parallel runways (26 and 27,
  34 and 35) as different directions.
- **Honest prior: small, below 0.1 s.** The main-lane tail is flights whose off-block lies far from the schedule
  (median 38 min against 10 min overall), which a configuration column does not identify.

## 1. The five columns (`research_370_konfig/features_konfig.py`, unit tests in `tests/test_features_konfig.py`)

All five read only `PHASE_mvt`, `ADEP_mvt`, `ADES_mvt`, `RUNWAY_mvt` and `MVT_TIME_UTC_mvt`, never a label, and are
computed per calendar month (the unit of the training files and of the ranking file, which holds January and July
2026 complete).

| column | definition |
|---|---|
| `kf_konfig` | arrival runways carrying at least 20 % of the arrivals of the departure's clock hour at its airport, sorted, joined with `+`; `(wenig)` below five arrivals; all runways of the hour if none reaches 20 %. Categorical. |
| `kf_seit_umkehr_min` | minutes since the airport's arrival flow last reversed, capped at 720 |
| `kf_bis_umkehr_min` | minutes until it next reverses, capped at 720 |
| `kf_peil_sin`, `kf_peil_cos` | sine and cosine of the initial great-circle bearing ADEP to ADES; missing if a code is unknown or both are equal |

**Flow.** An arrival lies on side A if its runway heading is less than 90 degrees from the airport's principal
arrival heading, else on side B. Principal headings, fixed from the most used arrival runway over the twelve 2025
training files (label-free, 29 September): EDDF 250, EDDM 260, EGLL 090, EHAM 180, LEBL 240, LEMD 320, LFPG 260,
LIRF 160, LSZH 140, LTFM 350. Fixed rather than recomputed per table, because at LFPG 26L and 08R carry 32 and 29 % of
arrivals and a per-table mode could flip the axis. An hour with at least five arrivals has flow A if at least half
lie on side A; a reversal is such an hour whose flow differs from the previous such hour, timed at that hour's first
arrival on the new side. Runway designators are normalised (`7` to `07`); `H`, `HEL` and `NA` (807 of 2,082,750 arrivals
in 2025) are ignored.

**Coordinates.** OurAirports `airports.csv` from `https://davidmegginson.github.io/ourairports-data/airports.csv`,
public domain ("All data is released to the Public Domain", ourairports.com/data, read 29 September 2026), listed at
12,733,496 bytes, last modified 29 September 2026 01:53:59 GMT. Saved to `.cache/research_370/airports.csv` (never
committed); its SHA-256 goes into the result. Codes are matched on `icao_code`, then `gps_code`, then `ident`; closed
airports are dropped.

**Label-free description, computed before this registration** (main-lane rows; no label read):

| | 2025 January and July (held out in 1_7) | 2026 January and July (scored) |
|---|---:|---:|
| rows | 339,046 | 339,551 |
| `kf_konfig` = `(wenig)` | 1.4 % | 1.4 % |
| distinct runway sets | 128 | 120 |
| within 60 min of a reversal | 6.3 % | 6.5 % |
| `kf_seit_umkehr_min` at the cap | 70.7 % | 71.4 % |
| LFPG east flow (`08R+09L`) | 29.5 % | 41.8 % |

17 runway sets of 2026 (1,581 rows) do not occur in 2025 January and July; in the screen, sets unseen in the learning
months enter as missing, as unseen categories of the other categorical columns do.

## 2. Arms and protocol (as `research_300/PREREG_weather.md` section 1)

- **real**: the 91 release features plus the five columns.
- **control 1, 2, 3**: the five columns permuted **jointly within airport and calendar month**, learning and
  held-out rows with independent seeds (1001 to 1003, 2001 to 2003). Same width as real, row-level link broken.
- **base**: the 91 features alone.

Same three training seeds for every arm, 177, 277 and 377. One LightGBM, learning rate 0.03, 255 leaves,
min_child_samples 50, subsample and colsample 0.8, **fixed 1,200 rounds**; main lane, target y minus ersatz_s,
prediction max(ersatz_s + model, 300). `kf_konfig` is categorical with the learning months' categories, like the
other categorical columns. Fifteen fits, one at a time under the shared heavy-work lock. Runner:
`research_370_konfig/konfig_screen.py`.

**Precondition, checked before any fit:** the bearing is present on at least 98 % of learning and of held-out rows.
Otherwise the runner stops without fitting and the coordinate matching is reviewed in an addendum.

## 3. Decision

Loss per held-out row = squared error averaged over the three seeds; the control loss is the mean over the three
controls. The columns **pass** only if all six hold:

1. real beats the mean control in **each** of the three seeds;
2. the gain over the mean control is positive in **both** held-out months;
3. it stays positive after removing its **100 most favourable rows**;
4. the 2,000-draw **airport-day block bootstrap** 95 % interval of that gain lies above zero;
5. the RMSE gain over the mean control is at least **0.3 s**;
6. **new against the weather screen:** the seed-averaged RMSE of real is below that of the base. Weather passed
   against its controls on 1_7 and then lost 0.137 s to the plain base on 3_9 (`PREREG_weather.md` addendum C); the
   gain over controls partly measures what noise columns cost, and what would ship is base plus columns.

Reported, not judged: gains per airport against control and base, LIRF, LTFM and LFPG pooled, rows within 60 min of
a reversal against the rest, coverage, the coordinate file's checksum.

## 4. Stages and what the outcome means

- **Stage 1: pair 1_7** (learning months 2 to 6 and 8 to 12, held out January and July; the repository's standard
  screen holdout, read by many earlier screens).
- **Stage 2, only if stage 1 passes: pair 3_9** (learning 2, 4, 5, 6, 8, 10, 11, 12; held out March and September;
  read once before, by the weather confirmation), same arms, seeds and six rules.

Both passing justifies a full-ensemble check under its own registration, not a candidate. Failing at either stage
closes runway configuration and destination bearing as main-lane columns.

## 5. Order

After the EDDF July confirmation (`research_360_integration/PREREG_eddf_july.md`), which holds the shared lock for
its extraction. The coordinate download waits for the owner's approval.

## Addendum A, 29 September 2026: coordinate file downloaded (before any fit)

Downloaded 10:56 local time with the owner's approval: 12,733,496 bytes, last modified 29 September 2026 01:53:59
GMT (both as listed in section 1), SHA-256 `e5b485bd1453002071231563dcc54fbf1efacb7c3cbaa486c73247b3aa1b2398`,
78,803 codes after matching. Bearing present on 99.68 % of the learning rows of pair 1_7 (1,723,531), 99.70 % of its
held-out rows (339,046) and 99.71 % of the scored 2026 main rows (339,551); the precondition (98 %) holds. Most
frequent unmatched destinations: ETNB, LSSB, OKBK, NTTT, APPH (20 to 36 codes per table). No label read.

## Addendum B, 29 September 2026: stage 1 (pair 1_7) fails all six rules

`konfig_screen.py 1_7`, `results/konfig_screen_1_7.json`. Seed-averaged RMSE: real 224.018, mean control 223.846,
base 223.653. Against the mean control **-0.171 s** (every seed negative: -41 / -128 / -61 MSE), January -91, July -65,
-235 without the best 100, airport-day [-206, +39]; against the base **-0.365 s**. Rows within 60 minutes of a flow
reversal lose most (-901 per row against control). By airport against control: LSZH +310, LEBL +260, EDDF +88, the
rest negative (LFPG -214, EDDM -202, EGLL -180, LEMD -179, EHAM -168, LTFM -163, LIRF -76). **Verdict: fail.** Stage 2
is not run; runway configuration and destination bearing as main-lane columns are closed.
