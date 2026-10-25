# Pre-registration: the corrector's capacity, selected on January/July, judged on four other months (pilot 15)

Written 24 October 2026, **before any fit for it**. Changes go into dated addenda.

## 0. Why

The corrector's settings (learning rate 0.03, 63 leaves, min_child_samples 100, subsample/colsample 0.8, 600 rounds) were fixed in pilot 5 and never tuned. Every corrector test since then used them.

## 1. Grid and selection

Rounds {600, 1200, 2400} x leaves {63, 255}; everything else unchanged (the default is 600 x 63). Selection on **January/July only** (339,046 rows, p1 = `ensemble_baseline`, pilot 6's day folds
`crc32(date) % 4`, mean of seeds 177 and 277): the configuration with the lowest RMSE. Ties within 0.05 s go to the smaller configuration (fewer rounds, then fewer leaves).

## 2. Judgement (the selected configuration against the default, on months the selection never read)

February/June (p1 from `.cache/ensemble_holdout_2_6`) and November/December (p1 from `.cache/ensemble_holdout_11_12`), each pair's own day folds `crc32(date) % 4` (the defaults are the cached arms pilot 12
`pair_26` and pilot 13 `own`), 643,659 rows together. Six rules, selected against default: 1. better in each seed; 2. positive in each of February, June, November, December; 3. positive without the 100 most
favourable rows; 4. airport-day bootstrap 95 % interval above zero; 5. RMSE gain at least 0.3 s; 6. better than the ensemble base. If the default is selected, nothing is judged and nothing changes.

## 3. Meaning

Pass: a second 2026 file `final_rebuild_2026_tuned.parquet` is built exactly like the current one (corrector trained on all January/July 2025 rows over the ensemble's predictions, seeds 177 and 277) with
the selected configuration, and becomes the preferred file. Fail: the current file stays.

## Addendum A, 24 October 2026: the default is selected; nothing judged

`pilot15_screen.py`, `results/pilot15_screen.json`, log `.cache/research_730/pilot15.log`. January/July selection RMSE: **600x63 200.992**, 1200x63 201.456, 600x255 201.477, 2400x63 202.566, 1200x255 203.126,
2400x255 205.747. Every larger corrector is worse; the default is kept and nothing is judged. Pilot 15b (`PREREG_corrector_capacity_small.md`) tests the smaller side.
