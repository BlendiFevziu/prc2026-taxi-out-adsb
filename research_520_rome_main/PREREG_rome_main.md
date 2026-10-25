# Pre-registration: a Rome copy classifier for the main lane

Written 30 September 2026, **before any fit**. Owner's request of the same night (build v71 with the label-mechanism
method). NM-free, challenge data only. No upload without the owner.

## Why

19 % of Rome's main-lane departures are copies (BLOCK = SCHED, so the label is the whole delay MVT - SCHED). A perfect
copy detector was worth 1.29e9 squared seconds on the 26,131 held-out January/July 2025 Rome main rows (about 7 s on
the official scale); the shipped gate reaches AUC 0.864 there (`experiments.csv`, 29 and 30 September). Rome's main
rows carry no ADS-B layer (26,516 base-owned rows in 2026). The fallback classifier of `research_510_rome_classifier/`
ranked copies better than the shipped model (0.898 against 0.850).

## Rows and reference

Test: the held-out January and July 2025 Rome main-lane rows of `harness.components_2025()` (26,131), reference the
pre-ADS-B base `layers.compose_base` (the same composition as v70's values on Rome main rows). Training: Rome main-lane
departures (AOBT_3 present) of the ten other 2025 months. For 2026 the classifier is refitted on all twelve months.

## Model, fixed now

LightGBM binary (copy = |label - delay| <= 60 s), 400 trees, learning rate 0.03, 31 leaves, min_child_samples 50,
lambda_l2 10, feature_fraction 0.9, bagging 0.8, seeds 177, 277, 377 averaged. Inputs present in the ranking data:
delay; AOBT_3 - SCHED; MVT - AOBT_3; EOBT_1 - SCHED; IOBT - SCHED; LOBT - SCHED; local-midnight crossing; local hour
of SCHED and of MVT; weekday; categoricals airline (FLIGHT_mvt[:3]), AIRCRAFT_OPERATOR_flt, destination country
(ADES_mvt[:2]), STAND_mvt, stand area (STAND_mvt[:1]), RUNWAY_mvt, AIRCRAFT_TYPE_flt, MARKET_SEGMENT_flt, FLIGHT_TYPE_flt.

## Arms

mix = p x delay + (1 - p) x base, floor 300. **A (decision): (base + mix) / 2.** B (reported): mix alone.

## Rules, all must hold for arm A

1. Pooled squared-error gain above zero.
2. Above zero without the 100 most favourable rows.
3. January and July both above zero.
4. All four half-months (days 1-15 and 16-31 of January and of July) above zero.
5. Planning: gain per row times 26,516, over 344,841, against v70's central 253.57, at least 0.2 s.

A pass leads to v71 = v70 plus arm A on the 26,516 Rome main rows, built and verified like v70.

## Result, 30 September 2026: fail on every rule

`rome_main.py`, `results/rome_main.json`. 133,085 training rows (copy share 0.208), 26,131 held-out rows (0.189).
**AUC 0.8638 against the shipped gate's 0.8638**: the new inputs (crossing, stand, operator, all flight-plan offsets)
add no ranking power in the main lane. Arm A -9.2e7 (-0.53 s planned), every half-month negative; arm B -3.1e8. The
shipped gate already carries this information, and adding a second copy weight double-counts it. No v71.
