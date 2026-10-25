# Pre-registration: Rome stage 2, the frozen partial-track model C1 at LIRF, unchanged

Written 29 September 2026 on branch `claude/stop-move-episodes`, **before any LIRF label is read at the layer level in
this repository and before any fit**. No fit, no new model. NM-free. The upload decision stays with the owner.

## What is known

- Stage 1 (`research_380_rome/stage1_coverage.py`, label-free): adsb.lol sees Rome in July 2025 (91.1 % of main-lane
  departures usable, median 33 ground points), December 2025 (60.0 %, 12) and January 2026 (65.8 %, 13; 6,767 partial
  rows), never in July 2026. December 2025 matches January 2026 on all 23 track features; July 2025 differs on three
  (denser tracks).
- **Codex's Rome screen** (`research_310/rome_partial_screen.py` in the frozen v56 checkout, result
  `rome_partial_screen.json`): a Rome-only residual model (200 x 7) trained on 2, 4, 18, 23 July 2025 and tested on
  9, 11, 16, 27 July gained +5,437 MSE per row, beat the context arm and five shuffles, was positive on every test day,
  and was **rejected** because the gain without its 100 most favourable rows was -0.2. Those eight days' LIRF labels
  have been read at the layer level; they are reported below descriptively only.

## Policy under test

The frozen C1 model of v62 (800 x 31, seeds 177/277/377, `layers.LAYERS["partial"]`, model hashes as recorded by
`research_320/results/capacity2.json`) applied at weight 1.0 with the 300 s floor to LIRF `partial_ground` rows. C1
never saw LIRF: its five airport flags (EDDF, EGLL, EHAM, LEBL, LSZH) are all zero there; `is_july` is 1. Inputs as in
v56: the held-out comparator (`layers.compose_base`), ersatz_s, schedule, gate, the hour phase from the movement time,
the 23 track summaries of the frozen extraction (`full_coverage_2026.extract_day`, via stage 1's cache).

## Confirmation block

LIRF main-lane partial rows of the **16 local July 2025 days not used by Codex's screen**: 3, 6, 7, 8, 10, 12, 13, 14,
15, 17, 21, 22, 25, 26, 29, 31 July. First half = the first eight of these dates, second half = the last eight. All
lie in the base's held-out months (Codex's base screen read every July label at the base level; disclosed).

## Rules (all must hold on the confirmation rows)

1. Pooled MSE gain of C1 over the base above zero.
2. Above zero after removing the 100 most favourable rows (the rule Codex's screen failed).
3. Above zero in each half.
4. Above zero within the degraded group (at most 10 ground points or a first point more than 800 m from the stand)
   and within the well-observed group.
5. The gain reweighted to LIRF's January 2026 partial mix (degraded 69.14 %, well observed 30.86 %, stage 1) above zero,
   and the planning value, 6,767 rows x that gain over 344,841 against v63's predicted 258.24, **at least 0.1 s at full
   transfer**.

Reported, not judged: Codex's eight days, strata, a within-day flight bootstrap, the per-day gains.

If all five hold, the 6,767 LIRF January 2026 partial rows get C1 in the final version, built and verified like v63.
If any fails, Rome stays at the base, and no Rome-specific model is started today.

## Result, 29 September 2026: fails rule 2, Rome stays at the base

`stage2_c1_transfer.py`, `results/stage2_c1_transfer.json`. Models as tested, reverse replay exact. 6,796 LIRF partial
rows on the 16 fresh days: RMSE 421.19 -> 417.99 s, +2,681 MSE per row (rule 1 holds), **-8,706 without the 100 most
favourable rows (rule 2 fails; those 100 rows carry 4.2 times the net gain)**, halves +2,162 / +3,191, degraded +2,197
and well observed +3,812, planning +0.10 s at full transfer. Seven of 16 days negative (-29,352 on 6 July to +50,799 on
13 July); within-day flight bootstrap [-1,868, +7,367]. Codex's eight days, descriptive: +2,367, -7,346 without the best
100. The same pattern as Codex's Rome-only model: the tracks rescue a few flights at Rome and cost a little on the rest.
**Verdict: fail.** As fixed above, Rome stays at the base and no Rome-specific model is started today.
