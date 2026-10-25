# Pre-registration: a second model family in the fallback lane (CatBoost), blended with the shipped fallback

Written 29 September 2026, **before any fit**. Owner's goal of the same day: look for multi-second gains; the fallback
lane (1.5 % of rows) carries about half of our squared error, and a leading contestant said his lead is mostly in the
rows without AOBT_3 and that "it completely depends on the models and training". A Discord post from another leading team:
"a second model family on the same inputs helped us more than new features". NM-free. No candidate, no upload.

## What is known

The shipped fallback (`research_240/fallback_anchor_production.py`, `prod_anchor`) is a LightGBM recipe: classifier
p for labels above 1.5 h (q = p^2.5), a height regressor on those rows (height = 0.7 x versp + 0.3 x model), a normal
regressor, and an anchored offset regressor on the neighbour-taxi anchor. Earlier fallback work changed inputs, gates,
anchors, calibration and selectors (all rejected or small); no record shows a different model family for these parts.
Today: count matching failed (-3.4 s), and no second copy mechanism or clock-offset pattern exists in the extreme rows.

## Arm

Every LightGBM part is replaced by a CatBoost counterpart on the same rows, inputs and targets, fixed now:
classifier CatBoostClassifier(600 iterations, depth 6, learning rate 0.05, l2_leaf_reg 10); height regressor 300
iterations; normal and anchored-offset regressors 400 iterations (same depth, rate and l2); categorical columns of
`main_model.KAT` as CatBoost categorical features (string values); random_seed 0, 4 threads. The CatBoost fallback is
combined exactly as the shipped one. **Arm = 0.5 x prod_anchor + 0.5 x CatBoost fallback.** The CatBoost fallback alone
is reported, not judged.

## Rules, six pairs pooled, all fallback rows (22,470), against the saved `prod_anchor`

1. Pooled MSE gain above zero.
2. Above zero after removing the 100 most favourable rows.
3. At least four of six pairs above zero.
4. January and July of pair 1_7 both above zero.
5. Official scale (5,290 fallback rows of 344,841, against v66's 256.4921): at least 0.1 s.

A pass leads to a 2026 build (all-year refit of the CatBoost parts, the shipped fallback reproduced first), verified
like the other candidates; the upload stays with the owner.

## Result, 29 September 2026: fails every rule

`fallback_family.py`, `results/fallback_family.json`. Fallback RMSE over six pairs: shipped 1,486.08, CatBoost alone
1,620.10, blend 1,517.97; -95,799 MSE per fallback row, **-2.86 s on the official scale**; -213,479 without the best
100; pairs 2_8 +108,360 and 4_10 +48,691, the other four negative (3_9 -394,283); January -23,166 and July -93,468 in
1_7. **Verdict: fail.** CatBoost with this recipe is the weaker family on this lane, alone and in the blend.
