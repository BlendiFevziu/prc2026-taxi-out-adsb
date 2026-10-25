# Pre-registration: a second partial-track model on sampling-aware features, averaged with C1

Written 29 September 2026, **before any feature is extracted and before any fit**. Owner's request of the same day
("let's try to reach 250 through changes before uploading again"). NM-free. No upload without the owner.

## Why

At Amsterdam January, averaging Codex's add-on with C1 beat the add-on alone by +4,521 MSE per row on ten fresh days,
every day positive (`research_480_weights/PREREG_weights.md`). The add-on uses different inputs: seven sampling-aware
track features (`robust_track_features.FEATURES` in Codex's checkout 8d98953) instead of C1's 23 summaries. Two models
with different inputs make different errors; C1 plus CatBoost on the same inputs did not (`research_430_robust/`). This
builds the add-on's kind of model for every airport C1 covers and averages it with C1.

## Features

Codex's extractor (`eham_recovery.extract`, 8d98953), generalised only in the airport it reads (`research_490_second/
extract_robust.py`): same matching, same stand maps as v56 (X-Plane first at EHAM, OSM elsewhere), same assertion that
every partial row replays its category and 23 summaries against the frozen v56 extraction exactly, then
`robust_track_features.summarise`. Airports EDDF, EGLL, EHAM, LEBL, LSZH. Days: C1's training set E; B1 and B2; the ten
fresh July and ten fresh January days downloaded today.

## Model

Codex's add-on recipe, unchanged: LightGBM regression, 200 trees x 7 leaves, learning rate 0.03, min leaf 80, L2 100,
row 0.8, column 0.9, max_bin 63, seeds 177, 277, 377; target y minus comparator; inputs `layers.CONTEXT` + the seven
features + the five airport flags + is_july (19). Trained on C1's training set E (partial rows, detector rows excluded).

## Arm and rules

Arm: max(base + (C1 + new) / 2, 300) against C1 alone, max(base + C1, 300), on partial rows of C1's cells.

1. Development B1: pooled gain above zero, both months above zero.
2. Development B2: pooled gain above zero.
3. Confirmation: partial rows of C1's cells on the twenty days downloaded today (ten July, ten January). **No fully
   unread January or July 2025 rows remain**: every day of both months is now held locally and used. These rows were
   read once today for a different question (July: the C1 + CatBoost blend and Heathrow July; January: C1 at weight
   1.25), never for this arm; disclosed. Rules: pooled gain above zero, above zero without the best 100, both months
   above zero, at least six of the eight C1 cells above zero (EDDF, LEBL, LSZH January and July; EGLL January; EHAM July), at least 70 % of days above zero.
4. Planning: the confirmation gain times the 2026 partial rows of the five airports' C1 cells, over 344,841, against
   v68's expected 255.89, at least 0.2 s.

A pass leads to 2026 feature extraction for those cells and a candidate built and verified like v68.

## Result, 29 September 2026: fail on every rule

`second.py`, `results/second.json`; comparator bit-identical in every block; 280 feature files, every partial row
replaying the frozen v56 extraction exactly. Trained on 20,044 rows of E (eight cells, detector rows excluded).

- B1: 19,605 rows, **-1,838** per row, both months, all eight cells and all fourteen days negative.
- B2: 11,038 rows, **-2,288**, all cells and days negative.
- Confirmation: 28,539 rows, **-1,832**, -2,227 without the best 100, all eight cells and **0 of 20 days** positive;
  planning **-0.88 s**.

**Fail.** Averaging halves C1's correction toward a much weaker 200 x 7 model on seven features. At Amsterdam January
the add-on won because C1 is not trained or deployed in that cell; in C1's own cells C1 is the stronger partner and the
average only dilutes it. Disclosure addendum: three of the ten July days (6, 8, 10 July) were also read earlier today
for the Frankfurt July confirmation (research_360), which the list above omitted; the result is negative either way.
