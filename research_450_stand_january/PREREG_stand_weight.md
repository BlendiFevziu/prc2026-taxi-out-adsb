# Pre-registration: the stand model at weight 1.0 instead of 0.5

Written 29 September 2026, **before any stand row is read at weight 1.0 and before the January tracks are complete**.
Owner's request to keep improving. NM-free. No upload without the owner.

## Why

The stand model (v57, 200 x 7, seeds 177/277/377) has always been applied at weight 0.5, fixed in `research_320/
PREREG_stand.md` and never tested against 1.0. The partial-track model's move from 0.5 to 1.0 (v58, `PREREG_shrink.md`)
gained 1.03 s. In v66 the stand model corrects 13,141 rows (9,749 in five v57 cells, 3,392 at LSZH and EHAM July).

## Blocks

- **Development:** the stand rows of v57's development and confirmation dates (`stand_experiment.STAGES`, 22 dates) in
  the five cells it is deployed in since v57 (EDDF January and July, EGLL January, LEBL January and July); these rows
  were read at weight 0.5 before, never at 1.0 (disclosed).
- **Confirmation (fresh):** stand rows never read at the layer level: EDDF July and LEBL July on the ten unread July days
  (1, 5, 6, 8, 10, 19, 20, 24, 28, 30 July; label-free counts 531 and 915) and EDDF, EGLL and LEBL January on the ten
  January days downloading now (3, 10, 11, 12, 15, 17, 22, 27, 29, 31 January). LSZH and EHAM rows are excluded: they
  were read today at weight 0.5 by the tests that put them into v66 or may put them into v67.

Detector-owned rows excluded; comparator bit-identical to the recomposed held-out base; floor 300 for both weights.

## Rules (gain = squared error at weight 0.5 minus squared error at weight 1.0)

1. Development: pooled gain above zero.
2. Confirmation: pooled gain above zero.
3. Confirmation: above zero after removing the 100 most favourable rows.
4. Confirmation: above zero in January and in July.
5. Planning: the confirmation gain per row times the stand rows the candidate corrects (13,141, plus the LSZH and EHAM
   January rows if `PREREG_stand_january.md` passes), over 344,841, against 256.4921: at least 0.1 s.

A pass leads to a candidate with the stand model at weight 1.0 on every row it corrects, built and verified like v66.

## Result, 29 September 2026: all five rules hold; candidate v67 built and verified

`stand_weight.py`, `results/stand_weight.json`. Development 3,099 rows +3,070 per row (every cell positive: EDDF/01
+1,497, EDDF/07 +3,765, EGLL/01 +3,285, LEBL/01 +2,622, LEBL/07 +3,382). Confirmation 2,984 fresh rows +3,519, +1,538
without the best 100, January +4,004 and July +3,003, every cell positive. Planning +0.34 s on 17,147 rows. **Pass.**

v67 (`build_v67.py`, `verify_v67.py`): v66 plus the stand model on the 4,006 base-owned LSZH and EHAM January stand rows,
and the stand model at weight 1.0 on all 17,147 rows it corrects. All 13,141 rows it already corrected reproduced
their v66 value exactly at weight 0.5 before the switch; the re-extraction reproduced Codex's features; 17,004 rows
change, RMS 56.2 s; everything else byte-identical to v66; 136 rows sit at the 300 s floor, where every definition gives
the same value, and are unchanged. **Prediction fixed before upload: central 256.12, range 256.35 to 255.93.**
