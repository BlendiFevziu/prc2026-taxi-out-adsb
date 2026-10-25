# Pre-registration: weight and capacity for the stand model

Written 28 September 2026 on branch `claude/stop-move-episodes`, **before any fit of this experiment**. Open data and
challenge data only, NM-free. No candidate, no upload. The stand model (v57, `PREREG_stand.md`) still runs as
deployed: 200 trees, 7 leaves, v56's eight training days (2,619 stand rows), weight s = 0.5. The partial-track
correction gained from full weight (v58) and from more capacity and days (v60, and `PREREG_capacity2.md`).

## Arms (36 columns including the guarded stand detector, seeds 177/277/377, floor 300)

| arm | training | trees × leaves | weight |
|---|---|---|---|
| reference (deployed) | S, 2,619 rows | 200 × 7 | 0.5 |
| K1 | E, 4,457 rows (S plus the six extra days of `PREREG_volume_capacity.md`) | 200 × 7 | 1.0 |
| K2 | E | 400 × 15 | 1.0 |
| K3 | E | 800 × 31 | 1.0 |

Everything else as deployed (learning rate 0.03, min leaf 80, L2 100, row 0.8, column 0.9, max_bin 63).

## Blocks, rows and limitation

Stand-observed rows in v57's five deployed cells (EDDF January and July, EGLL January, LEBL January and July): B1
(development, 1,997 rows) and B2 (replication, 1,102 rows), the same days as before. Both were read by the stand
test, so B2 is a blocked replication, not an untouched confirmation.

## Rules (three comparisons with the reference, Bonferroni 1 − 0.05/3 on B1)

Survive B1 only if: pooled gain above zero; January and July each above zero; above zero without the 100 most
favourable rows; adjusted airport-day interval above zero; each paired seed above zero; at least three of five cells
above zero. Survivors must beat their context-only and five permutation controls (4101 to 4105) at the same
training, capacity and weight. Replication on B2, frozen models, the same rules adjusted for the survivors. If more
than one passes, the largest **B1** gain is proposed. Planning: v61's 9,749 eligible 2026 stand rows per cell times
the B2 gain per row, over 344,841, against 258.9401; **0.1 s at full transfer** is the bar.

## Result, 28 September 2026: every arm passes development, none passes replication

`stand_capacity.py`, `results/stand_capacity.json`; 4,457 training rows, B1 1,997 rows, B2 1,102 rows. Gain per
stand row against the deployed model:

| arm | B1 (97.5 %) | B1 w/o best 100 | B2 (98.3 %) | B2 w/o best 100 | verdict |
|---|---|---|---|---|---|
| K1 E, 200 × 7, s = 1 | +3,331 [2,056, 4,660] | +505 | +4,359 [2,112, 6,484] | **−471** | fail |
| K2 E, 400 × 15, s = 1 | +4,794 [3,024, 6,642] | +674 | +6,930 [4,320, 9,547] | **−36** | fail |
| K3 E, 800 × 31, s = 1 | +5,072 [3,185, 7,011] | +392 | +7,403 [4,893, 9,964] | **−462** | fail |

All seeds, both months and at least four of five cells positive on both blocks; controls lose 5,669 to 8,048. The
single failing rule is the same for all three: on 1,102 replication rows the gain rests on the 100 most favourable
(9 % of the rows). As registered, nothing passes and no candidate follows. The rule is not relaxed after the fact.
A fair retest would need a larger untouched block; the seven 2025 days not held locally break the embargo with the
training days, so none exists now.
