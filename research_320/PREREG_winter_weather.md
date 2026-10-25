# Pre-registration: winter weather in the main lane, tested on unread winter months

Written 28 September 2026 on branch `claude/stop-move-episodes`, **before the first fit**. Owner's choice: "Test winter
weather first" (before uploading v58). Open data only (cached IEM METAR, challenge rows). No candidate, no upload.

## Evidence so far

- `research_300/PREREG_weather.md`: eight METAR columns passed the equal-width screen on pair 1_7 (January +499 MSE per
  row against controls, +0.255 s against the 91-feature base), then **failed** the confirmation on pair 3_9 (March and
  September; 0.137 s worse than the base). The effect looked like a winter one.
- `research_320/leads_diagnostic.py` (January/July 2025 residuals of production, within month): beyond an airport-month
  offset, **January visibility +93 MSE per row, icing risk +29, freezing +26**, frozen precipitation +3; temperature,
  dew-point spread and wind negative; July nothing.

## Hypothesis and features (fixed now)

Four winter columns of `research_240/features_weather.WEATHER` improve main-lane predictions in winter months:
**`wx_visibility_km`, `wx_frozen`, `wx_freezing`, `wx_icing_risk`** (latest cached METAR at or before the movement, at
most 3 h old). Temperature, dew-point spread and wind are left out on the diagnostic's evidence.

## Design

- **Pairs 2_8 and 6_12**, never used for weather: learning on the ten other months each (the repository's six-pair
  design, `research_250/main_time_cv.py`), held out February and August, June and December.
- **Tables**: the release's all-year table (`.cache/clock_queue_release_20260912/data/months_1_2_..._12.parquet`) filtered
  by month. Checked before this registration: it is not byte-identical to tables built from the pair's months alone;
  about 1 % of rows differ in context features at month boundaries (turnaround, arrival counts, sequence and queue),
  because the all-year table sees the neighbouring months' movements. Those are label-free movement data; every arm
  uses the same rows, so the comparison between arms is unaffected.
- **Arms**, same rows, one LightGBM as the earlier weather screens (learning rate 0.03, 255 leaves, min_child_samples 50,
  subsample and colsample 0.8, fixed 1,200 rounds): **real** (91 release features + the four columns), **control 1 to 3**
  (the four columns permuted jointly within airport and calendar month; learning seeds 1001-1003, held-out 2001-2003),
  **base** (the 91 features). Seeds 177, 277, 377. 30 fits, one at a time under the shared lock.
- **Judged on the winter months only**: February (pair 2_8) and December (pair 6_12), main lane. August and June are
  reported, not judged.

## Rules (all must hold)

1. real beats the mean control on the winter rows in each seed;
2. February and December each positive against the mean control;
3. positive against the mean control after removing the 100 most favourable rows;
4. airport-day block bootstrap (2,000 draws) 95 % interval against the mean control above zero;
5. **real beats the base** on the winter rows (the 3_9 lesson: beating noise columns is not enough);
6. **minimum meaningful**: winter MSE gain against the base at least 200 per row (about +0.1 s overall if it carried to
   the January half of the 2026 scored set).

## Decision

Pass: winter weather justifies a production refit with the four columns (most of a day), which is the owner's decision;
v58 would be uploaded first or together, as the owner chooses. Fail: main-lane weather is closed, and v58 is uploaded
as it is if the owner confirms.

## Addendum A, 28 September 2026: fail, main-lane weather is closed

30 fits, 23:56 to 01:23 local, under the shared lock (`winter_weather_screen.py`, `results/winter_weather.json`).
306,650 winter rows (February and December). Real against the mean control +46.2 MSE per row (seeds -5.0 / +90.1 /
+53.6), **February -157.5**, December +223.4, **-106.1 without the best 100 rows**, airport-day 95 % [-233.0, +319.8];
against the 91-feature base +136.5, below the 200 minimum. Five of six rules fail. Summer months, reported only:
August -95.5, June +64.1 against the base. The January signal of the 1_7 screen does not carry to other winter months.
As registered, main-lane weather as features is closed; v58 goes up as it is if the owner confirms.
