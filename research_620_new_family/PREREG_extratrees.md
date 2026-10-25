# Pre-registration: a second model family (ExtraTrees) blended into the base where the base is final

Written 30 September 2026, **before any fit**. Owner: "do the first one" (a stronger base for the rows with no ground
data). Motivation: a leading team's note that "a second model family on the same inputs helped us more than new
features"; every feature screen at LTFM/LFPG/LIRF failed; the base mixes LightGBM, CatBoost and neural sequence nets but
no bagged, randomised trees. NM-free, challenge data only. No upload without the owner.

## Model, fixed now

sklearn ExtraTreesRegressor: 200 trees, max_features 0.33, min_samples_leaf 40, bootstrap with max_samples 0.5,
random_state 0. Inputs: the 91 v45 main-lane features (`candidate_features.main_features(MERKMALE_HAUPT)`) from the
release cache `.cache/clock_queue_release_20260912/data`; categoricals (`main_model.KAT`) as ordinal codes fitted on the
training rows (unseen -1), missing numbers as -999. Target: label minus ersatz_s, clipped to [-3600, 3600]; prediction
ersatz_s plus the model's value.

## Test: learning months to January and July, weight chosen across months

Fit on the ten learning months' main rows (`months_2_3_4_5_6_8_9_10_11_12`), predict the held-out January and July main
rows (339,046, aligned by ID with `harness.components_2025`). Blend max((1 - w) x base + w x ExtraTrees, 300), base =
`layers.compose_base`. w is chosen on January from {0.05, 0.10, 0.15, 0.20, 0.30} by squared error and applied to July,
and the other way round. Rules, all must hold: gain above zero in both directions; above zero without the best 100 in
both; at least seven of ten airports above zero (both directions pooled); planning: per airport the pooled gain per row
times its 2026 main rows whose v77 value is still the base, over 344,841, against v77's official 246.3837, at least
0.1 s.

A pass leads to v78: the model refitted on all twelve months, the mean of the two chosen weights, applied only to 2026
main rows whose v77 value is exactly the base (every layer's rows keep their values).

## Result, 30 September / 1 October 2026: fail on every rule

`extratrees.py`, `results/extratrees.json`. ExtraTrees alone 275.17 s RMSE on the 339,046 held-out rows against the
base's 214.69; error correlation 0.78. Best weight 0.05 in both directions, and still January-picked on July -336 per
row, July-picked on January -154; all ten airports negative; planning -0.31 s. A randomised bagged family on the same
91 inputs adds no information the boosted ensemble lacks. Nothing built.
