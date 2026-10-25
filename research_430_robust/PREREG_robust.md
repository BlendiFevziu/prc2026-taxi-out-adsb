# Pre-registration: a drift-robust partial-track correction (movement-only inputs, a second model family)

Written 29 September 2026, **before any fit**. Owner's request of the same day, after methodology hints from a leading
team on Discord: validate January against July and back; check every input for 2025 to 2026 drift (adsb.lol's receiver
network changed, so inputs describing where an aircraft was heard do not transfer, inputs describing its movement do);
correct a strong base; a second model family on the same inputs. NM-free. No candidate, no upload.

## Why

C1 (800 x 31, 35 inputs, seeds 177/277/377) now corrects about 99,000 scored rows. Its realised leaderboard gain has been
0.59 (v62), 0.72 (v63) and 1.16 (v64) of the validated value; some of the loss is plausibly input drift.

**Drift, label-free, fixed now** (2025 partial rows of the E, B1 and B2 extraction tables, 61,388, against Codex's 2026
feature table, eight C1 cells weighted by their 2026 rows): standardised mean shift ground_points 0.449, stand_min_m
0.420, stand_first_m 0.377, stand_median_m 0.320, then sampled_distance_m 0.241 and all others at most 0.202. **Inputs
with a shift of at least 0.30 SD are dropped in the movement-only arms: ground_points, stand_min_m, stand_first_m,
stand_median_m** (all four describe where the aircraft was heard). 31 inputs remain.

## Arms (training set E, 24,093 rows; target y minus comparator; seeds 177, 277, 377, mean correction; weight 1, floor 300)

- **C1**: the frozen models (reference).
- **M**: LightGBM with C1's recipe on the 31 inputs.
- **F**: (C1 + CatBoost on the 35 inputs) / 2. CatBoost fixed now: 800 iterations, depth 6, learning rate 0.05,
  l2_leaf_reg 10, RMSE loss, 4 threads, random_seed = the seed.
- **MF**: (M + CatBoost on the 31 inputs) / 2.

## Rules, per arm, against C1 (all must hold)

1. **Development B1** (`volume_capacity.judge` with alpha 0.05 / 3): pooled gain, both months, positive without the best
   100 rows, adjusted interval above zero, every seed, at least five of seven cells.
2. **Replication B2**: the same judge.
3. **Cross-season** (the new gate): the arm and C1 are each refitted on E's January rows only and judged on B1 and B2's
   July rows, then on July rows only and judged on January rows; the arm must beat the C1 refit pooled in **both**
   directions.
4. **Planning:** the B2 gain against C1, times the 99,059 scored 2026 rows C1 corrects in v64, over 344,841, against
   257.0986, at least 0.1 s.

Reported, not judged: the gains reweighted to the 2026 track-quality mix, per cell, per stratum. If several arms pass,
B1 decides the choice. A passing arm replaces C1 in a candidate built and verified like v64; the upload stays with the
owner.

## Result, 29 September 2026: no arm passes; F fails only the concentration rule

`robust.py`, `results/robust.json`; E 24,093, B1 16,029, B2 8,894 rows; frozen C1 hashes checked.

| arm | B1 gain | B2 gain | without best 100 (B1 / B2) | cross-season 01->07 / 07->01 | planning | verdict |
|---|---:|---:|---:|---|---:|---|
| M (movement-only) | -443 | -281 | -852 / -864 | -308 / -59 | -0.16 s | fail, every rule |
| F (C1 + CatBoost) | +447 | +356 | -46 / -249 | +337 / +834 | +0.20 s | fail, rule "without best 100" in B1 and B2 |
| MF | -49 | -42 | -631 / -860 | +25 / +571 | -0.02 s | fail |

F holds every other part of the judge in both blocks (both months, adjusted interval above zero: B1 [178, 733], B2
[39, 720]; every seed; at least five of seven cells) and the cross-season gate, but its gain rests on about a hundred
rows. Dropping the four drifting inputs costs in both blocks and both season directions: in this pipeline the inputs
describing where an aircraft was heard carry signal that transfers across seasons. **No candidate.**
