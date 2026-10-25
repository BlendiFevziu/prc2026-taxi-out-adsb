# Pre-registration: the frozen C1 model in three unused cells (Munich January, Madrid July, Heathrow July)

Written 29 September 2026 on branch `claude/stop-move-episodes`, **before any label of these cells is read at the layer
level and before any fit** (there is none: the model is frozen). Owner's request of the same day. NM-free. The upload
decision stays with the owner.

## Why

The frozen C1 correction (800 x 31, seeds 177/277/377, weight 1.0, floor 300; `layers.LAYERS["partial"]`, models as
recorded in `research_320/results/capacity2.json`) earned 0.60 s in v63 from one newly switched-on cell, EDDF July
(11,993 rows; `research_360_integration/PREREG_eddf_july.md`). Five airport-months with good 2026 tracks still get no
correction. This registration covers the three that have same-month 2025 tracks to test on:

| cell | 2026 partial rows | 2026 median ground points | 2026 degraded share | 2025 test rows from |
|---|---:|---:|---:|---|
| EDDM/01 | 6,802 | 125 | 13.07 % | all 21 local January 2025 days |
| LEMD/07 | 8,692 | 31 | 53.83 % | all 24 local July 2025 days |
| EGLL/07 | 9,048 | 81 | 6.19 % | 6, 8, 10 July 2025 |

Degraded = at most 10 ground points or a first point more than 800 m from the stand (research_360 part 1). 2026 values
are label-free, from Codex's 62-day feature table (EGLL) and the coverage audit's cache (EDDM, LEMD).

**What has been read before (disclosed).** C1 never used EDDM or LEMD rows; their airport flags are all zero, as at
Rome, where the same model failed (`research_380_rome/PREREG_stage2.md`: +2,681 pooled, -8,706 without the best 100).
EDDM January labels were read by the Munich stand-detector tests on stand-observed rows (a different category) and by
base screens; LEMD labels by base screens only. At EGLL, every local July day except 6, 8 and 10 has been used by
layer training or evaluation, so only those three are fresh; they were extracted for the EDDF July confirmation, whose
script read only EDDF rows.

## Rows and inputs

EDDM and LEMD: the frozen extraction (`full_coverage_2026.extract_day` at 9884150, OSM stands, the same call the 2026
coverage audit made) on the local days, main-lane departures; comparator, ersatz_s, schedule, gate from the held-out
2025 components (`harness.components_2025`, `layers.compose_base`); hour phase as v56; airport flags zero; is_july as
the month. EGLL: the cached frozen v56 extraction of 6/8/10 July (`stand_experiment.extract(..., "eddf_july_confirm")`),
comparator bit-identical to the recomposed base. Only `partial_ground` rows.

## Rules, per cell, all must hold

1. Pooled MSE gain of C1 over the base above zero.
2. Above zero after removing the 100 most favourable rows.
3. EDDM and LEMD: above zero in each half (EDDM: 1, 2, 4, 5, 6, 7, 8, 9, 13, 14 January against 16 to 30 January;
   LEMD: 2 to 14 July against 15 to 31 July). EGLL: above zero on each of the three days.
4. Above zero within the degraded and within the well-observed group; a group with fewer than 30 test rows is not
   judged.
5. The gain reweighted to the cell's 2026 mix (a group under 30 test rows takes the cell's pooled gain) above zero, and
   the planning value, the cell's 2026 rows x that gain over 344,841 against v63's 258.0008, **at least 0.1 s at full
   transfer**.

Each cell is judged on its own. A passing cell enters v64 (v63 plus C1 on that cell's base-owned 2026 partial rows),
built and verified like v63; a failing cell stays at the base. Madrid January and Munich July, which have no same-month
2025 tracks, are not part of this registration.

## Result, 29 September 2026: EDDM January and LEMD July pass, EGLL July fails

`cells_confirm.py`, `results/cells_confirm.json`; models as tested, reverse replay exact in every cell.

| cell | test rows | RMSE base -> C1 | gain per row | without best 100 | halves or days | groups (degraded / well) | planning | verdict |
|---|---:|---|---:|---:|---|---|---:|---|
| EDDM/01 | 5,014 | 158.3 -> 108.5 s | +13,278 | +8,213 | +14,718 / +11,946 | +4,956 / +14,960 | +0.52 s | **pass** |
| LEMD/07 | 2,777 | 189.0 -> 165.3 s | +8,405 | +829 | +11,401 / +6,083 | +4,839 / +10,403 | +0.36 s | **pass** |
| EGLL/07 | 233 | 330.8 -> 293.3 s | +23,437 | -20,679 | +38,910 / +4,383 / +474 | +22,291 / (27 rows) | +1.19 s | fail (rule 2) |

EDDM January is robust: all 21 days positive, the best 100 rows carry 39 % of the gain. LEMD July passes narrowly: the
best 100 carry 90 %, three of 24 days negative. EGLL July has only 233 thin 2025 rows on its three fresh days, so
removing 100 of them decides; it stays at the base. On the January 2025 days with Munich detector estimates, the
partial rows the detector does not own gain +14,750 per row against +15,332 on its own (descriptive), so leaving the
detector's rows untouched in 2026 costs nothing per row.

v64 (`build_v64.py`, `verify_v64.py`): v63 plus C1 on the 14,089 base-owned partial rows of the two cells (EDDM 5,397
after the 1,405 rows the Munich detector owns; LEMD 8,692). The 2026 context recipe reproduced 3,000 rows of Codex's
context table bit for bit before use. 14,083 rows change, mean -18.4 s; everything else byte-identical to v63.
**Prediction fixed before upload: central 257.44, range 257.69 to 257.22.**
