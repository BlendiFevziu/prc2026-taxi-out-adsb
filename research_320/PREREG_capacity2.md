# Pre-registration: a second capacity step for the partial-track correction

Written 28 September 2026 on branch `claude/stop-move-episodes`, **before any fit of this experiment**. Open data and
challenge data only, NM-free. No candidate, no upload. Reference release: v61 (258.9401), which carries arm 4 of
`PREREG_volume_capacity.md` (14 days, 400 trees, 15 leaves, s = 1.0) on 72,979 rows.

## Why

Going from 200 × 7 to 400 × 15 gained +3,362 MSE per eligible row on B1 and +3,940 on B2, and +1.28 s officially
(v60). The larger model is still small. This asks whether the curve keeps rising.

## Arms (training set E, 24,093 rows; 35 columns, seeds 177/277/377, s = 1.0, floor 300; everything else as arm 4)

| arm | trees | leaves |
|---|---|---|
| reference: arm 4 | 400 | 15 |
| C1 | 800 | 31 |
| C2 | 1,600 | 63 |

Learning rate 0.03, min leaf 80, L2 100, row 0.8, column 0.9, max_bin 63 stay fixed.

## Blocks and limitation

B1 (development) and B2 (replication) as in `PREREG_volume_capacity.md`, same rows, same guarded-detector exclusion.
Both were read by the first capacity study, so B2 is a blocked replication, not an untouched confirmation. The seven
2025 days not held locally all lie within one day of an E training day and would break the two-day embargo, so no
fresh block exists.

## Rules (same as the first study; two comparisons, Bonferroni 1 − 0.05/2 on B1)

An arm survives B1 against arm 4 only if: pooled gain above zero; January and July each above zero; above zero
after removing the 100 most favourable rows; adjusted airport-day interval above zero; each paired seed above zero;
at least five of seven cells above zero. Survivors must beat their context-only and five permutation controls
(3101 to 3105) fitted at the same capacity. Replication on B2 with frozen models, the same six rules, adjusted for
the number of survivors. If both arms pass, the one with the larger **B1** gain is proposed; B2 decides pass or fail
only. Planning: 2026 eligible rows per cell in v61 times the B2 gain per row, over 344,841, against 258.9401;
**0.1 s at full transfer** is the bar for proposing a separately reviewed candidate.

## Result, 28 September 2026: C1 (800 × 31) passes, C2 (1,600 × 63) fails replication

`capacity2.py`, `results/capacity2.json`. Gain per eligible row against arm 4 (the model in v61):

| arm | B1, 97.5 % interval | cells / dates | B2, 95 % interval | cells / dates | verdict |
|---|---|---|---|---|---|
| C1 800 × 31 | +1,059 [655, 1,556], w/o best 100 +461 | 7 of 7 / 14 of 14 | +1,415 [1,034, 1,813], w/o best 100 +438 | 7 of 7 / 8 of 8 | **pass** |
| C2 1,600 × 63 | +1,003 [248, 1,796], w/o best 100 +21 | 5 of 7 / 11 of 14 | +1,564, w/o best 100 **−11** | 6 of 7 / 7 of 8 | fail |

Seeds positive for both arms on both blocks; context and permutation controls lose 11,469 to 13,808 against arm 4.
The curve peaks at C1: C2 gains on the pooled mean but loses once the 100 most favourable rows are removed, the
signature of fitting a few rows. Planning for C1 on v61's 72,979 eligible rows: 299.7 MSE per scored row, **+0.58 s
at full and +0.51 s at v60's 0.88 transfer** (about 258.43 from 258.94); the 0.1 s bar holds.
Label-free support: on 72,946 scored 2026 rows C1 moves the correction by 26.8 s RMS against arm 4 (B2: 27.8 s),
correlation 0.976 (B2: 0.972); quantiles match. Next, separately reviewed: candidate v62 = v61 with C1 in place of
arm 4 on the same rows, not built yet.
