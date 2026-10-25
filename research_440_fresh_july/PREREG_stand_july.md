# Pre-registration: the frozen stand model at Zurich July and Amsterdam July, on ten unread July 2025 days

Written 29 September 2026, **before any stand-observed row of these cells and days is read at the layer level and
before any fit** (there is none: the model is frozen). Owner's request to keep going. NM-free. No upload without the
owner.

## Why

The v57 stand model (200 x 7, seeds 177/277/377, weight 0.5, floor 300, `layers.LAYERS["stand"]`, 36 inputs including
the guarded detector's taxi time) corrects stand-observed rows in five cells. **Zurich** was in its deployable set and
gained in development (LSZH January +6,794 per row on 578 rows) and confirmation (LSZH +5,538), but v56's drift rule
removed both LSZH cells at build time (January first-point distance 1.01 SD, July last-point distance 1.26 SD), the
same gate that withheld EDDF July's partial cell before today's fresh confirmation passed. **Amsterdam's**
stand-observed rows outside the Amsterdam detector (2,280 in July 2026) were never in its scope; the model was trained
with stand rows of all five airports, EHAM included. In 2026, base-owned stand-observed rows: LSZH July 1,112, EHAM July
2,280 (label-free, Codex's feature table and the v62 ownership).

## Rows and days

Stand-observed (`observed_at_assigned_stand`) main-lane rows of LSZH July and EHAM July, detector-owned rows excluded,
on the ten local July 2025 days no layer has read for these cells: 1, 5, 6, 8, 10, 19, 20, 24, 28, 30 July (6/8/10 were
read only for EDDF and EGLL partial rows). Rows and inputs from the cached frozen v56 extractions (`fresh_july`,
`eddf_july_confirm`); comparator bit-identical to the recomposed held-out base. Label-free counts: LSZH 703, EHAM 696.
First half 1, 5, 6, 8, 10; second half 19, 20, 24, 28, 30.

## Rules, per cell (weight 0.5, as deployed)

1. Pooled MSE gain over the base above zero.
2. Above zero after removing the 100 most favourable rows.
3. Above zero in each half.
4. Above zero on at least seven of the ten days.

**Joint rule 5**, over the cells passing 1 to 4: their 2026 base-owned rows times their pooled gain, over 344,841,
against v65's 256.6145, at least **0.05 s** at full transfer. The usual 0.1 s is lowered here, before any result,
because the two cells hold only 3,392 scored rows; a cell's own planning value must still be positive.

Passing cells enter a candidate (v65 plus the stand model on those rows), built and verified like v65.

## Result, 29 September 2026: both cells pass; candidate v66 built and verified

`stand_july.py`, `results/stand_july.json`; comparator bit-identical, reverse replay exact. LSZH July: 703 rows, +4,471 MSE
per row, +1,460 without the best 100, halves +4,297 / +4,639. EHAM July: 696 rows, +5,562, +1,358 without the best 100,
halves +5,341 / +5,777. Both at least seven of ten days positive. Joint planning 0.10 s (bar 0.05). **Pass.**

v66 (`build_v66.py`, `verify_v66.py`): v65 plus the stand model at weight 0.5 on the 3,392 base-owned stand rows (LSZH
1,112, EHAM 2,280; 1,441 Amsterdam detector rows untouched); the 2026 re-extraction reproduced Codex's features exactly;
3,362 rows change, RMS 36.1 s; everything else byte-identical to v65; NM-free. **Prediction fixed before upload: central
256.53, range 256.58 to 256.49.**
