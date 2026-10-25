# Rebuild file for January, February, June and July 2026 (17 October 2026)

Owner's own project after the competition's close; **not uploaded**, no leaderboard, no probe-derived value.

`python3 -u -B research_730_adsb_learner/build_2026_final.py table` then `... build`; output `.cache/research_730/rebuild_2026/final_rebuild_2026.parquet`
(never committed), SHA-256 `acca1ea3c76e7adf2d80016aeaed9a9d18000f23944d4a45201a3c1029d093d5`, report `.cache/research_730/rebuild_2026/build_report.json`,
corrector models `corrector_s177.txt`, `corrector_s277.txt`.

**Recipe.** Stage 1: the frozen four-month base (`.cache/research_800/base_final_predictions.parquet`: v45 all-year ensemble with the sequence nets, anchored fallback,
copy gate, Rome rules; main lane equal to the scored v47 recipe). Stage 2: the ADS-B corrector (12 track + 6 stand columns, `features_adsb.py`), trained on all 339,046
January/July 2025 main-lane rows over the same recipe's held-out predictions, two seeds averaged, applied to all 661,713 main-lane rows of the four 2026 months with
that month's adsb.lol tracks (119 days; 11 June 2026 has no archive). Fallback rows (9,077) keep the base. Checks: format equal to `final_submitting.parquet`, all finite,
fallback rows identical to the base.

**Evidence.** Pilot 6 (+15.8 s on this ensemble, January/July 2025, day folds) and pilot 7 (+15.2 s when trained on one 2025 month pair and applied to another). Track
coverage 2026: January 82.8 %, February 84.5 %, June 75.5 %, July 81.8 % of main-lane rows (2025 training months: 79.7 %). Correction on 2026: mean -6.9 s, mean absolute 54.7 s
(2025 out of fold: -7.5 s, 47.2 s), so the corrector behaves on 2026 as it did when validated.

**Not measurable.** No 2026 label is available, so the file has no score. Against the submitted final file the January/July rows also drop v80's later layers (partial-track
corrections, stand model, Amsterdam add-on), which the corrector replaces. Watch: EGLL June/July (mean absolute correction about 103 s) and LEMD (2026 coverage 97 % against
78 % in the 2025 training months).

**Not in it:** the daily reconciliation (pilot 8b failed rule 4), learned stands (deferred), a fallback-lane model with ADS-B (too few matched rows).
