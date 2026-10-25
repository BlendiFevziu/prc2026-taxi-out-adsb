# Pre-registration: Heathrow July stand rows, Amsterdam detector rows, Paris July partial rows

Written 29 September 2026, **before any fit and before any label named below is read for these questions**. Owner's
request of the same day ("let's do all three"). NM-free. No upload without the owner.

## A. The stand model at Heathrow July (10,242 base-owned stand rows in 2026): excluded, label-free

Heathrow July 2025 has almost no stand-observed rows (85 on the ten unread July days), so only a sister-month
transfer is possible, under the rule that admitted Munich July and excluded Madrid January (`research_400_cells/
PREREG_cells2.md` rule 2: v56's drift rule between the sister month's 2025 rows and the cell's 2026 rows). Computed
label-free before this file: EGLL July 2026 base-owned stand rows against the 1,501 EGLL January 2025 stand rows of the
stand model's development, confirmation and fresh January days flag **three of the 23 track features**
(ground_points 1.70 SD, last_still_age_s 1.15, near_stand_fraction 1.04). **Rule 2 fails; no test, no candidate.**

## B. The stand model and C1 instead of the raw Amsterdam detector

In v67 the Amsterdam stand detector's value is shipped unchanged on its rows (2026: 905 + 1,441 stand-observed and 792 +
1,050 partial rows). Hint from a leading team: keep the rows where the record contradicts the sensor; a correction that
takes the detector as an input can learn when not to trust it.

Rows: the 1,732 EHAM detector rows of the 27 held-out 2025 days with the production detector's value
(`research_360_integration/prepare_detectors_2025.py`; B1, B2, 5/7/19 January, 15/17 July), rows and inputs from the
frozen v56 extractions. Never read at the stand-model or C1 level (EHAM detector rows were excluded from both); read at
the detector level by v55's tests (disclosed). Comparator bit-identical to the recomposed held-out base.

- **B1**, stand-observed rows (995), both months: the stand model at weight 1.0 (max(base + correction, 300)) against
  the detector's value.
- **B2**, July partial rows: C1 (weight 1.0) against the detector's value. January partial rows are not tested.

Per sub-part, all must hold: pooled MSE gain over the detector above zero; above zero without the 100 most favourable
rows; at least 60 % of days above zero; B1 also above zero in January and in July. Jointly over the passing
sub-parts: their gain per row times their 2026 rows (B1 2,346, B2 1,050) over 344,841 against v67's 256.0858, at least
**0.05 s**, each positive.

## C. The frozen C1 at Paris July (1,896 base-owned partial rows in 2026)

Rows: LFPG July 2025 main-lane partial rows on all 31 July days (C1 never used LFPG; its airport flags are zero, as at
Madrid, where it passed, and Rome, where it failed), from the frozen extraction (`full_coverage_2026.extract_day`, OSM
stands) as in `research_400_cells/cells_confirm.py`. Rules of `PREREG_cells.md`: pooled gain above zero; above zero
without the best 100; each half (1 to 16 July, 17 to 31 July) above zero; degraded and well-observed groups above zero
(groups under 30 rows not judged); the gain reweighted to LFPG July's 2026 mix, times 1,896 over 344,841 against
256.0858, **at least 0.1 s**.

Passing parts enter one candidate, built and verified like v67.

## Result, 29 September 2026: no part passes

`three.py`, `results/three.json`. **A** excluded label-free as fixed above. **B1** (995 stand-observed detector rows):
detector RMSE 94.2 s, stand model 136.7 s, -9,810 MSE per row, -16,767 without the best 100, 7 % of days positive,
January -24,400 and July -1,284: fail. **B2** (451 July partial detector rows): detector 60.8 s, C1 66.8 s, -767 per
row, -2,947 without the best 100, 46 % of days positive: fail. At Amsterdam the detector is accurate enough that a
correction only adds noise; the hint to keep contradicting rows does not carry over here. **C** (1,070 LFPG July
partial rows): 283.3 -> 269.3 s, +7,711 per row, but **-10,437 without the best 100** and planning 0.08 s (bar 0.1):
fail; halves +19,368 / +2,399, degraded +8,149 (1,008 rows), well observed +589 (62). No candidate.
