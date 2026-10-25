# Pre-registration: Frankfurt July with the frozen partial-track model, confirmed on three fresh days

Written 29 September 2026 on branch `claude/stop-move-episodes`, **before the tracks of 6, 8 and 10 July 2025 are
downloaded and before any label of those days is read**. NM-free. No fit, no new model. No upload.

## Why

v56's release drift gate withheld EDDF July because its 2026 track features differ from training in their means.
Part 1 (`PART1_RESULT.md`) found that the difference is a change in the mix of track quality: 31 % of EDDF July
partial tracks were degraded in 2025, 73 % in 2026. On 13 held-out July days of 2025 the frozen C1 model (v62's
partial-track correction) gains in every track-quality stratum with at least 30 rows; reweighted to the 2026 mix it
keeps 8,843 MSE per row, about +0.3 to +0.6 s on the 11,993 withheld rows. This registration tests that on days no
layer has ever been evaluated on, and deliberately replaces the mean-based drift gate for this one cell by
stratified evidence.

## Policy under test

v62, plus the frozen C1 model (800 × 31, seeds 177/277/377, weight 1.0, floor 300, `layers.LAYERS["partial"]`)
applied to EDDF July `partial_ground` rows. EDDF has no direct detector, so no row is excluded. Nothing else changes.

## Track quality, fixed as in part 1

Density: ground points at most 10 / 11 to 40 / more than 40. Start: distance of the first ground point from the
assigned stand at most 200 m / 200 to 800 m / more than 800 m / no stand. **Degraded** = at most 10 points or a
first point more than 800 m out. 2026 EDDF July mix, label-free from Codex's 2026 feature table: **degraded
72.98 %, well observed 27.02 %** (sparse and late alone 39.64 %).

## Development evidence (already inspected, descriptive only)

13 held-out July days of `PART1_RESULT.md`, 6,761 rows: pooled +16,038 MSE per row (183.7 → 133.1 s), degraded
+11,047, well observed +18,312, reweighted to the 2026 mix +8,843.

## Confirmation block (fresh)

EDDF July `partial_ground` rows of **6, 8 and 10 July 2025**, tracks from each day's preferred adsb.lol release
(`research_300/fetch_preferred.py`), rows and features from the frozen v56 extraction (`stand_experiment.extract`),
base = the held-out comparator (`layers.compose_base`, bit-identical to the extraction's column). Never used at the
layer level; all three lie in the held-out months of Codex's base screen, which read every January and July label
at the base level (disclosed). Each is at least two days from every C1 training day (2, 3, 4, 18, 21, 22, 23, 29 July).

## Rules (all must hold on the confirmation rows)

1. Pooled MSE gain of C1 over the base above zero.
2. Above zero after removing the 100 most favourable rows.
3. Above zero on each of the three days.
4. Above zero within the degraded group and within the well-observed group.
5. The gain reweighted to the 2026 mix (0.7298 × degraded + 0.2702 × well observed) above zero, and the planning
   value, 11,993 rows × that gain over 344,841 against 258.5995, **at least 0.1 s at full transfer**.

Group level rather than stratum level, because three days hold only about 40 sparse-and-late rows. Three days give
three blocks, too few for a day-block interval; a within-day flight bootstrap is reported as descriptive only.

If all five hold: candidate v63 = v62 plus C1 on the 11,993 EDDF July rows, built and independently verified
separately; upload only with the owner's decision. If any fails: EDDF July stays withheld, and augmentation for
sparse, late tracks is registered separately or not at all.

## Result, 29 September 2026: all five rules hold (pass)

`eddf_july_confirm.py`, `results/eddf_july_confirm.json`. Tracks from the preferred releases (6 July prod-0, 8 and 10
July staging-0; receipts in `research_300/results/fetch_preferred_2025-07-{06,08,10}.json`). Base bit-identical to
the extraction's comparator. 1,633 EDDF July partial rows (560, 520, 553).

| | value |
|---|---|
| RMSE base / with C1 | 153.10 / 102.58 s |
| gain per row, pooled (rule 1) | +12,915 MSE |
| without the 100 most favourable rows (rule 2) | +4,726 |
| per day, 6 / 8 / 10 July (rule 3) | +10,037 / +17,965 / +11,082 |
| degraded (509 rows) / well observed (1,124) (rule 4) | +11,938 / +13,358 |
| reweighted to the 2026 mix (rule 5) | +12,321, planning +0.83 s full, +0.41 s half |
| within-day flight bootstrap 95 % (descriptive) | [10,666, 15,337] |

**Verdict: pass.** One warning, descriptive: sparse and late tracks, 39.6 % of 2026 rows, lose on these days
(-3,147 on 36 rows; +5,649 on 183 development rows). Weighting each stratum with at least 30 confirmation rows by its
own confirmation gain and its 2026 share (smaller strata at zero) gives 6,081 per row, **+0.41 s at full and +0.20 s
at half transfer**. As fixed above, the next step is candidate v63 = v62 plus C1 on the 11,993 EDDF July rows, built
and verified separately; no upload without the owner.
