# Pre-registration: CatBoost as a second family for C1+ and K2+, on the fresh days

Written 1 October 2026, **before any fit**, under `POLICY_tail_rule.md`. NM-free. No upload without the owner.

## Arms, fixed now

CatBoost (800 iterations, depth 6, learning rate 0.05, l2_leaf_reg 10, RMSE, seeds 177/277/377, 4 threads) fitted on
exactly the rows the deployed model was fitted on (E + B1 + B2): for C1+ the partial rows on C1's 35 inputs, for K2+ the
stand rows on the 36 stand inputs; target label minus base. Arm: max(base + (deployed + CatBoost) / 2, 300) against
max(base + deployed, 300).

## Test

The twenty fresh days (F), partial rows of C1+'s eight cells and stand rows of K2+'s nine cells, Amsterdam detector
rows excluded, comparator bit-identical. Disclosed: F decided C1+ against C1 and K2+ against K2, and its July rows
scored the old C1 + CatBoost blend on 29 September; it has never scored these blends. Rules per family, all must hold:
pooled gain above zero; airport-day bootstrap 95 % lower bound above zero; January and July above zero; at least two
thirds of the cells above zero; every paired seed above zero; planning (C1+ 84,970 rows, K2+ 17,147 rows) against
v77's official 246.3837, at least 0.1 s for C1+, 0.05 s for K2+.

A passing family enters v78 on the deployed model's rows.

## Result, 1 October 2026: both families pass

`catboost_layers.py`, `results/catboost_layers.json`. C1+ with CatBoost: 28,539 fresh partial rows, +352 per row,
day bootstrap 95 % [154, 601], January +510, July +225, planning
+0.18 s. K2+ with CatBoost: 5,862 fresh stand rows, +1008 per row, [663, 1414],
January +1326, July +671, planning +0.10 s. Both enter v78.

## Candidate v78, built and verified (1 October 2026)

`build_v78.py`, `verify_v78.py`: all 84,970 C1+ rows and 17,147 K2+ rows reproduced v77 with the deployed models first;
101,248 change (mean +0.5 s, RMS 18.8 s); everything else identical to v77; the verifier recomputed every value through
independent input routes with the base from the pre-NM file. SHA 7c17917f... Prediction fixed in the build: central
246.16, best 246.05, worst 246.30 (from v77's official 246.3837).
