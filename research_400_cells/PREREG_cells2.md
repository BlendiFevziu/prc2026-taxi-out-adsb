# Pre-registration: the frozen C1 model in the two cells without same-month 2025 tracks (Madrid January, Munich July)

Written 29 September 2026, before any candidate is built. Owner's request of the same day. NM-free. No upload without the
owner.

## The situation, and what this registration can and cannot show

Neither cell has 2025 tracks in its own month (LEMD January 2025: 0 ground points a day; EDDM July 2025: about 7), so no
fresh label-based confirmation is possible. Each cell's sister month passed `PREREG_cells.md` (EDDM January +13,278 MSE
per row on 21 days, every day positive; LEMD July +8,405, narrowly). The partial-track correction has passed in both
months at every airport where both were tested (EDDF, LEBL, LSZH). This registration decides from label-free transfer
checks built on those already-read results; **no new label is read**, and that limit is stated here on purpose.

The checks below were computed label-free while writing this file (before any build):

| | LEMD January 2026 | EDDM July 2026 |
|---|---|---|
| partial rows / base-owned | 12,478 / 12,478 | 9,539 / 7,525 (2,014 owned by the Munich detector) |
| v56 drift rule against the sister month's test rows | **10 inputs flagged** (ground_points 2.84 SD, sampled_distance_m 2.50, stand_median_m 2.06, last_moving_age_s 2.03, last_ground_age_s 2.02, ...) | **none flagged** |

## Rules, per cell, all must hold

1. The sister month passed `PREREG_cells.md`.
2. The v56 drift rule (no input shifting by more than one SD of the sister month's test rows, no missingness change above
   0.20) holds between the sister month's 2025 test rows and the cell's 2026 base-owned partial rows.
3. Every track-quality stratum holding at least 5 % of the cell's 2026 rows has a positive sister-month gain measured
   on at least 30 rows.
4. The sister month's per-stratum gains (strata under 30 rows at zero), weighted by the cell's 2026 mix, times its
   base-owned rows over 344,841 against v64's 257.0986: **at least 0.1 s at full transfer**.

Application as before: weight 1.0, floor 300, base-owned partial rows only. **LEMD January fails rule 2 and stays at the
base.** EDDM July is judged by rules 1 to 4 in `build_v65.py` before anything is written.

## Result, 29 September 2026: Munich July passes rules 1 to 4; Madrid January stays at the base

`build_v65.py` evaluated the rules before writing: sister month passed, no drift flag, every stratum with at least 5 %
of the 2026 rows positive on at least 30 sister-month rows, weighted gain 12,969 per row, planning +0.55 s at full
transfer. v65 = v64 plus C1 on the 7,525 base-owned EDDM July partial rows (2,014 detector rows untouched); 7,519 change,
mean -2.4 s; everything else byte-identical to v64; independently verified (`verify_v65.py`). **Prediction fixed before
upload: central 256.66, range 256.93 to 256.44.** No same-month confirmation exists for this cell; that risk is the
reason for the wide range.
