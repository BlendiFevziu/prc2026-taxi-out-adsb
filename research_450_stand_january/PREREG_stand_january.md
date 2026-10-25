# Pre-registration: the frozen stand model at Zurich January and Amsterdam January, on ten unread January 2025 days

Written 29 September 2026 while the tracks download, **before any label of these days is read and before any fit**
(there is none: the model is frozen). Owner's request of the same day. NM-free. No upload without the owner.

## Why

The same model passed at Zurich July and Amsterdam July on ten fresh July days (every day positive in both cells;
`research_440_fresh_july/PREREG_stand_july.md`) and went into v66 (256.4921). Zurich January was in v57's scope and gained
in development (+6,794 per row), then was removed by v56's drift rule (first-point distance 1.01 SD). In 2026,
base-owned stand-observed rows: LSZH January 2,869, EHAM January 1,137 (905 more belong to the Amsterdam detector).

## Rows and days

Stand-observed main-lane rows of LSZH January and EHAM January, detector-owned rows excluded, on the ten January 2025
days not held locally before today and so never read at the layer level: 3, 10, 11, 12, 15, 17, 22, 27, 29, 31 January
(3, 11 and 27 were base-only test days in Colab set A; every January label was read at the base level by Codex's base
screen; disclosed). First half 3, 10, 11, 12, 15; second half 17, 22, 27, 29, 31. Rows and inputs from the frozen v56
extraction (`stand_experiment.extract`), comparator bit-identical to the recomposed held-out base.

## Rules

Exactly those of `PREREG_stand_july.md`: per cell, pooled gain above zero, above zero without the 100 most favourable
rows, both halves above zero, at least seven of ten days above zero (weight 0.5, floor 300); jointly over the passing
cells, their 2026 base-owned rows times their gain over 344,841 against v66's 256.4921 at least 0.05 s, each cell's own
planning value positive. Passing cells enter a candidate (v66 plus the stand model on those rows), built and verified
like v66.

## Result, 29 September 2026: both cells pass

`stand_january.py`, `results/stand_january.json`. LSZH January: 856 rows, +5,131 MSE per row, +1,115 without the best
100, halves +4,859 / +5,361, all ten days positive. EHAM January: 623 rows, +6,886, +717 without the best 100, halves
+6,134 / +7,524, all ten days positive. Joint planning 0.128 s. **Pass.** (The chained start failed because the watcher
waited on a process pattern its own command line contained; the test was started by hand at 19:13, after the download
had completed and before anything was read.)
