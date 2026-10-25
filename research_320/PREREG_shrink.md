# Pre-registration: full weight (s = 1.0) instead of 0.5 for the ADS-B residual corrections

Written 27 September 2026 on branch `claude/stop-move-episodes`, **before the confirmation dates below were downloaded
and before any label on them was read**. Idea 1 of the owner's list (coverage-conditioned blending), in the form the
exploration supports. Open data only; the models are frozen, only the weight of their correction changes. No upload.

## Exploration (development dates, spent)

`blend_headroom.py`, `results/blend_headroom.json`. Prediction = max(comparator + s x correction, 300):

| model | rows | s = 0.5 (production) | s = 1.0 | best (s = 1.25) |
|---|---|---|---|---|
| v56 partial-track, its 8 development-test dates, 7 deployed cells | 9,467 | +5,018 | +7,640 | +8,055 |
| v57 stand, its 14 development dates, 5 kept cells | 1,997 | +6,029 | +8,710 | +8,798 |

Every evidence stratum (ground points, still fraction, first-observation age) and every cell prefers s from 1.0 to 2.0:
the corrections are systematically under-weighted, as the NM add-on's were (s* = 1.237 in v52). Conditioning on
evidence adds little, so one global weight is tested. **s = 1.0 is fixed now**, below the development optimum, because a
larger weight also enlarges any 2026 drift the development dates cannot show.

## Components, both judged separately

- **A: v56's partial-track correction**: the three real-seed models (byte-identical to Codex's), `partial_ground` rows
  in v56's seven deployed cells, without the guarded Amsterdam detector's rows. Controls: v56's five permutation-control
  model sets (shuffle 3101 to 3105), their track columns permuted within airport and date exactly as in v56's screen.
- **B: v57's stand correction**: the three real-seed stand models, stand rows in v57's five kept cells. Controls: its
  five permutation-control model sets (shuffle 4101 to 4105).

Per row, gain = squared error at s = 0.5 minus squared error at s = 1.0, same frozen corrections. A control's gain is
the same quantity for its own corrections.

## Confirmation dates (fresh)

Drawn with `numpy.random.default_rng(20260928)` over the January and July 2025 days no study has touched (every date of
research_290, 300, 310 and 320, the reserve included, excluded), first four of each month: **January 5, 19, 23, 7 and
July 15, 29, 21, 17, 2025**. A date that cannot be fetched completely is replaced by the next of its month's permutation
(January 10, 3, 11, ...; July 6, 24, 10, ...).

## Rules, per component

1. pooled gain above zero;
2. above each of the five controls' gains;
3. January and July each above zero;
4. above zero after removing the 100 most favourable rows;
5. airport-day block bootstrap (2,000 draws) 95 % interval above zero;
6. each seed's own correction gains from the higher weight;
7. at least five of seven cells (A) or three of five (B) above zero.

**Minimum meaningful gain**: the passing components together, per cell gain x 2026 rows v56 or v57 changes in that cell,
divided by 344,841, must reach 0.3 s at full transfer. A component that fails keeps s = 0.5.

If a component passes, candidate v58 = the scored v57 with that component's rows recomputed at s = 1.0, every other row,
including every NM, Munich, Amsterdam and fallback row, byte-identical to v57, verified independently before the owner
decides on an upload. Nothing is re-tuned on these dates.

## Addendum A, 27 September 2026: component A passes, component B fails

All eight dates fetched from their preferred release and verified (three July days only on the second attempt, after
an IPv6-only network could not reach GitHub; the same dates, no replacement). `shrink_experiment.py`, 23:41 local,
`results/shrink_confirm.json`.

**A, v56's partial-track correction: pass, all seven rules.** 9,245 rows, s = 1.0 against 0.5: +2,496 MSE per row,
+1,402 without the best 100, airport-day [1,715, 3,487], January +2,727, July +2,227, all seven cells positive
(+536 to +5,923), seeds +2,487 / +2,467 / +2,495; the five controls scaled the same way lose 61 to 112, so the gain
comes from real signal.
**B, v57's stand correction: fail**, on rule 4 alone: +3,245 pooled but -502 after removing the best 100 rows. It keeps
s = 0.5.

Planning with A only: 516.9 MSE per scored row, **+0.99 s at full transfer, +0.50 s at half**; the 0.3 s minimum holds.
Next, as registered: candidate v58 = the scored v57 with v56's partial-track rows recomputed at s = 1.0, everything
else byte-identical, verified independently before the owner decides.

## Addendum B, 28 September 2026: the official result

The owner authorised the upload after the winter-weather screen failed ("upload v58"). `upload_v58_once.py` uploaded the
candidate once at 06:38:42 UTC as `upbeat-goblin_v58.parquet` (remote size equal to local, 4,727,687 bytes). **Official
score 259.9035** against v57's 260.9292: a gain of **1.0257 s**, slightly better than the full-transfer end of the range
fixed before the upload (259.94). Realised 534.2 MSE per scored row against 516.9 predicted (1.03). Record:
`results/upbeat-goblin_v58.json`. Nothing was tuned from this score.
