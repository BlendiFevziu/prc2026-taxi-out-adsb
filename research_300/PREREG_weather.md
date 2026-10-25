# One focused main-lane weather experiment: METAR columns against equal-width permutation controls

Written 25 September 2026 on branch `claude/stand-geometry`, after the geometry test reached its decision (addendum B
of `PREREG_geometry.md`) and **before the first fit**. Nothing below may change after the first fit; changes go into
a dated addendum. Private research: no production change, no candidate, no upload, no merge.

## 0. What is known, and what is not

- Main lane: the only earlier evidence is a residual diagnostic on v45's January/July holdout, bin corrections by
  weather state that removed at most 211 total MSE (81 cross-fitted), below that study's own 500 bar
  (`research_240/main_weather_diagnostic.py`). Weather was never given to a main-lane model as features.
- Fallback lane: weather was screened and rejected (`research_240/fallback_weather_screen.py`). Not repeated here.
- The single-seed main-lane screen confounds a small feature effect with the column count (STATUS.md, 23 September,
  as corrected by Codex): a candidate must be compared with controls of the same width, over several training seeds.

## 1. Hypothesis and arms

**Hypothesis:** the eight METAR columns of `research_240/features_weather.WEATHER` (temperature, dew-point spread,
wind, gust excess, visibility, frozen precipitation, freezing, icing risk) carry row-level information about the
main-lane taxi time beyond the 91 release features: de-icing, low-visibility procedures and wind.

- **real**: the 91 release features plus the eight columns; the latest cached IEM METAR report at or before the
  movement time, at most 3 h old (`.cache/research_240/weather`, public domain).
- **control 1, 2, 3**: the same eight columns permuted **jointly within airport and calendar month**, learning and
  held-out rows with independent seeds (1001-1003, 2001-2003). Same width as real; the row-level link is broken.
- **base**: the 91 features alone. Descriptive only (a different width), never the comparator.

Every arm is fitted with the **same three training seeds, 177, 277 and 377**. Protocol otherwise the repository's
main-lane screen: one LightGBM, learning rate 0.03, 255 leaves, min_child_samples 50, subsample and colsample 0.8,
learning months 2-6 and 8-12, held out 1 and 7, main lane, target y - ersatz_s, prediction max(ersatz_s + model,
300); but a **fixed 1,200 rounds** (no early stopping on the held-out months, so no held-out label picks a model).
Fifteen fits, each under the shared heavy-work lock, one at a time. Script: `research_300/weather_screen.py`.

## 2. Decision

Loss per held-out row = squared error, averaged over the three seeds; the control loss is the mean over the three
controls. The weather columns **pass** only if all hold:

1. real beats the mean control in **each** of the three seeds;
2. the gain is positive in **January and in July**;
3. the gain stays positive after removing its **100 most favourable rows**;
4. the 2,000-draw **airport-day block bootstrap** 95 % interval of the gain lies above zero;
5. the RMSE gain against the mean control is at least **0.3 s**, the repository's bar for main-lane screens.

Reported, not judged: per airport, the base arm, the weather coverage.

## 3. What the outcome means

Failing closes main-lane weather as features, as fallback weather was closed before. Passing justifies a six-pair
confirmation and a full-ensemble check, not a candidate. The January/July 2025 holdout is the repository's standard
screen holdout and has been read by earlier screens; this is a screen, not a fresh validation.

## Addendum A, 25 September 2026: the result

Fifteen fits, 16:16 to 16:59 UTC, each under the shared lock (`weather_screen.py`, `results/weather_screen.json`),
learning 1,723,531 rows, held out 339,046; METAR present on 100 % of rows.

| Seed | real | control 1 / 2 / 3 | base (91, descriptive) |
|---|---|---|---|
| 177 | 223.657 | 224.122 / 224.295 / 224.038 | 223.708 |
| 277 | 223.382 | 223.717 / 223.878 / 223.764 | 223.704 |
| 377 | 223.168 | 224.062 / 223.967 / 223.820 | 223.560 |

Seed-averaged: real **223.402**, mean control 223.963, base 223.657. **All five rules hold: pass.** Gain against the
mean control 0.561 s (250.8 MSE per row; per seed 221.7, 181.0, 349.7); January +499.1, July +48.4; +95.3 without
the best 100 rows; airport-day 95 % [97.2, 392.6]. Per airport: LFPG +998, EDDF +589, EDDM +524, LSZH +290,
EGLL +252, LEBL +231, LEMD +203, EHAM -5, LTFM -109, LIRF -530.

**How to read it.** The pre-registered comparison is against equal-width controls, and it passes. Against the
91-feature base the real weather columns gain **0.255 s**; the rest of the 0.561 s is the cost that eight noise
columns impose on the controls. The effect is a January (winter) effect, small in July, and negative at LIRF and
LTFM. As fixed in section 3: this justifies a six-pair confirmation and a full-ensemble check, not a candidate.

## Addendum B, 25 September 2026: the confirmation, staged; written before its first fit

Section 3 calls for a six-pair confirmation. Only two pairs have prepared main-lane tables (the release's
`months_1_7` and `months_3_9` with their learning tables); the other four would have to be built from scratch, several
hours of heavy work on a shared machine. So the confirmation runs in two stages, each with section 1's arms, seeds,
1,200 rounds and section 2's five rules unchanged (the month check applies to the pair's two held-out months):

- **Stage 1: pair 3_9** (learning months 2, 4, 5, 6, 8, 10, 11, 12; held out March and September), using the release's
  prepared tables. Output `results/weather_screen_3_9.json`.
- **Stage 2, only if stage 1 passes**: the four remaining pairs (2_8, 4_10, 5_11, 6_12), tables built with the
  repository's table builder, judged pooled over all six pairs plus at least four of six pairs positive.

If stage 1 fails, the weather screen's pass on 1_7 is not confirmed and main-lane weather stops there.

## Addendum C, 25 September 2026: stage 1 fails, so the confirmation stops here

Pair 3_9, fifteen fits 17:47 to 18:29 UTC under the shared lock (`weather_screen.py 3_9`,
`results/weather_screen_3_9.json`); learning 1,378,679 rows (months 2, 4, 5, 6, 8, 10, 11, 12), held out 344,852;
METAR present on 100 % of rows.

| Seed | real | control 1 / 2 / 3 | base (91, descriptive) |
|---|---|---|---|
| 177 | 201.583 | 201.639 / 201.711 / 201.830 | 201.390 |
| 277 | 201.576 | 201.587 / 201.560 / 201.647 | 201.584 |
| 377 | 201.400 | 201.449 / 201.663 / 201.424 | 201.175 |

Seed-averaged: real **201.520**, mean control 201.612, base 201.383. Gain against the mean control **0.092 s**
(37.2 MSE per row; per seed 57.9, 8.6, 45.0). Rule 1 holds; the other four fail: March -38.7 and September +105.4,
-66.9 without the best 100 rows, airport-day 95 % [-46.8, +116.5], and 0.092 s against the 0.3 s bar.
**Verdict: fail.** Against the 91-feature base the real weather columns lose 0.137 s.

As fixed in addendum B, stage 2 is not run, the 1_7 pass is not confirmed, and main-lane weather as features stops
here. What the two pairs show together: the effect found on 1_7 is a January effect (January +499.1 MSE per row
there), not a general one. Whether a winter-only effect would replicate is a different hypothesis. It could only be
tested on pairs whose held-out months have not been read for weather (2_8 and 6_12), under its own pre-registration,
and it is not started without the owner.
