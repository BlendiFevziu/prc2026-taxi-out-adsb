# Pre-registration: CatBoost as second family inside the Amsterdam January add-on

Written 1 October 2026, **before any CatBoost fit and before any test row is scored with it**. Owner: "ok run the eham
test". Rule set: the tail-rule policy (`research_640_tail_rule/POLICY_tail_rule.md`). NM-free. No upload without the owner.

## Training rows, verified before writing this

Codex's add-on (`.cache/codex_eham/eham_recovery/real_{177,277,377}.pkl`) was fitted on the EHAM partial rows of his
eight TRAIN days (`S.MS.TRAIN`). Rebuilt here from `S.extract(MS.TRAIN, "train")` plus the cached EHAM track features:
3,752 rows (his report says 3,752), comparator bit-identical to `layers.compose_base`, and a LightGBM refit with his
settings reproduces all three seed models exactly (max difference 0.0 s). No test label was read for this check.

## Model, fixed now

CatBoostRegressor, 800 iterations, depth 6, learning rate 0.05, l2_leaf_reg 10, RMSE, seeds 177/277/377, thread_count 4,
on the same 14 inputs (`layers.eham_inputs`) and the same target (y minus comparator). Same recipe as the CatBoost family
in v78/v79.

## Arm

Deployed in v80 on the 9,752 EHAM January rows: max(base + (add-on + C1+) / 2, 300). Candidate: the add-on replaced by
(add-on + CatBoost) / 2, so max(base + ((add-on + CatBoost) / 2 + C1+) / 2, 300). Gain = deployed error minus candidate
error, squared.

## Test rows

Fresh January (`weights.JAN10`, the ten days of `S.extract(JAN10, "fresh_january")`), EHAM/01 partial rows without a
detector value, as in `research_550_review/c1plus_other_cells.py` decision b: 4,686 rows. These rows have been read
before (add-on weight, the C1 and C1+ averages) but never with a CatBoost model, the same standing as the reused rows in
v76 and v79. Reported in addition: the development days (`weights.DEV_EHAM`).

## Rules, all must hold (fresh January)

Pooled gain above zero; airport-day bootstrap (5,000 draws, seed 0) 95 % lower bound above zero; at least seven of ten
days above zero; every seed pair above zero; development pooled gain above zero; planning: gain per row times 9,752 over
344,841, against v80's official 245.046, at least 0.02 s. The bar is lower than the usual 0.05 s because the expected
effect was stated to the owner beforehand as about 0.03 s (0.01 to 0.06 s); it is recorded here so it cannot move.

A pass leads to v81 = v80 with the candidate formula on the 9,752 rows (gate: each row first reproduces its v80 value).

## Result, 1 October 2026: pass; v81 built and verified

`eham_cat.py`, `results/eham_cat.json`. Training rows 3,752, Codex's three seeds refit exactly. Fresh January (4,686
rows): +573 per row, bootstrap 95 % [+364, +811], 10 of 10 days, seeds +558/+561/+591; development (5,098 rows) +583,
11 of 11 days. Planning +0.033 s. `build_v81.py`, `verify_v81.py`: all 9,752 rows reproduced v80 first; 9,683 change
(RMS 11.8 s); the verifier recomputed every value with each model's own feature names. SHA 649b9bab... Prediction fixed
in the build: central 245.020, best 245.006, worst 245.036. Not uploaded.
