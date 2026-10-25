# Pre-registration: the ADS-B features on all rows, including the fallback lane (pilot 4)

Written 15 October 2026, **before the all-rows table is built and before any fit**. Same conditions as the earlier pilots (private research, no
leaderboard, no upload). Changes go into dated addenda.

## 0. Why

Pilots 1 and 2 used only main-lane rows (`hat_aobt == 1`, an `AOBT_3` clock exists). The scored ranking set holds all rows. Rows without `AOBT_3` (about 1.6 % of
departures in January and July 2025, 5,373 rows) have no network off-block clock at all, so ADS-B is the only direct evidence of the push-back for them, and they
carry a large share of the squared error in our own earlier audits. This pilot asks whether the 12 + 6 ADS-B and stand columns help on all rows, and where.

## 1. Rows, arms, protocol

Rows: all 344,419 January and July 2025 departures of `months_1_7.parquet` with a label (main lane and fallback), the same release features as before (the 91 columns,
`hat_aobt` among them). Folds, seeds, hyperparameters and 1,200 rounds as in pilots 1 and 2; the target is `y - ersatz_s`, prediction `max(ersatz_s + model, 300)` (for rows without
`AOBT_3`, `ersatz_s` is the release's own substitute, unchanged). Arms: **base** (91), **real** (91 + 12 ADS-B + 6 OSM stand columns), **control_k** (the 18 permuted jointly within airport and day).

## 2. Decision (all must hold), judged on rows with a label up to 20,000 s

The decision rows exclude the date-shifted and Rome giant records (label above 20,000 s, a handful per month), which no ADS-B column addresses and which dominate the
all-rows RMSE by chance. 1. real beats the mean control in each of the two seeds; 2. positive in both months; 3. positive after removing the 100 most favourable rows;
4. airport-day block bootstrap 95 % interval above zero; 5. RMSE gain over the mean control at least **0.3 s**; 6. real beats the base.
Reported, not judged: the same figures on all rows, and separately on the fallback rows (`hat_aobt == 0`) and the main-lane rows, by airport.

## 3. Meaning

Pass: the ADS-B columns go into the all-rows model, and the fallback-lane gain tells us whether the lane deserves its own ADS-B model. Fail on the fallback rows only (main lane
passes as before): keep the columns for the main lane and look at the fallback lane separately.

## Addendum A, 15 October 2026 (before any fit): matching the fallback rows

The first table build matched **0 of 5,373** fallback rows: they have no flight-table record, so `CALLSIGN_flt` is empty on all of them (checked on all twelve months: 22,470 fallback
rows, all without a callsign, 99.7 % with `FLIGHT_mvt`). The matcher therefore needs a callsign for them. Rule: for a row with `FLIGHT_mvt` and no `CALLSIGN_flt`, the derived callsign is
the ICAO airline prefix of its two-letter code plus the digits of the flight number (`LH982` to `DLH982`); the prefix map is learned from all 2025 departures that have both and whose callsign ends with
the flight digits (79.5 % of them), taking each two-letter code's most frequent ICAO prefix; it reads no label. Rows whose derived callsign has no track stay unmatched. The same derivation also fills
the 0.9 % of main-lane rows without a callsign. Nothing else in the registration changes.

## Addendum B, 15 October 2026: the run is invalid for the fallback rows; the main-lane block replicates pilots 1 and 2

`pilot4_screen.py` printed verdict "fail", with NaN everywhere except the main-lane block. Cause: for all 5,373 fallback rows `ersatz_s` (`MVT - AOBT_3`) is undefined, so the main-lane
target (`y - ersatz_s`) and prediction (`ersatz_s + model`) are NaN there; the registration's "the release's own substitute" does not exist. The fallback lane is a separate model in this project
(different target, fewer inputs) and cannot be tested with the main-lane machinery. **The verdict "fail" is void**: it does not test the hypothesis. Valid output: main-lane rows with a label up to 20,000 s
(339,045), real (91 + 12 ADS-B + 6 stand columns) 220.41, mean control 236.43, base 236.61, gain **+16.0 s** over the controls and **+16.2 s** over the base (pilots 1 and 2 together: +15.7 s, so
this replicates them). Fallback match share after the callsign derivation: 7.6 % (about 410 of 5,373 rows). Not repeated: the fallback lane needs its own model first, and ADS-B
reaches too few of its rows to test with two months of tracks; revisit when four or more months of tracks are held.

## Addendum C, 24 October 2026: the fallback lane revisited with twelve months of tracks held; closed, ceiling too small

Exploration, no test (January/July, fallback p1 = the September candidate's held-out prediction, `table_1_7_all.parquet`): 5,373 fallback rows, RMSE 1,756.9; 410 matched to a track (7.6 %), and those
hold **0.5 %** of the lane's squared error (their RMSE 432.1, median absolute error 156 s). The lane's error sits in the giant records (99th percentile of the label 11,049 s), which no track reaches. Even
a perfect ADS-B model on the matched rows would be worth under 0.1 s overall, so the lane gets no ADS-B model. More months of tracks do not change the match share, which is set by the missing callsigns.
