# research_260: ordered neighbour sequence model (14 September 2026, the team's clone)

Research pass run from a fresh clone on a Mac mini (M4, 10 cores, 24 GB, macOS 26, Python 3.12 in a
project-local `uv` venv on external storage, torch 2.8 with MPS). Branch `claude/research-240-kevin`,
started from `claude/research-240-stage-stand-order` at commit 9796128 (the current baton), not from the
`codex/v45-clock-queue-handoff` commit the prompt named, because that branch already contained the v46
release (268.2394 s) and the 13 September screens. Nothing here was uploaded, merged, or built into a model
bundle. Official best remains v46.

## Replication in the fresh clone

| Check | research_250 record | This clone |
|---|---:|---:|
| January/July fallback baseline (v44 recipe, research encoding) | 1759.277 | 1759.277 |
| January/July single LightGBM, 91 columns, seed 177 | 223.698 (1,179 rounds) | 223.698 (1,179 rounds) |
| Main-lane rows, train / held out | 1,723,531 / 339,046 | 1,723,531 / 339,046 |
| Unit tests | 69 pass | 69 pass |

## 1. Extreme-row anchor census (closed)

`extreme_anchor_census.py`, descriptive, all twelve 2025 source files. The unexplained fallback extremes
(no AOBT_3, label above 5,400 s, not a schedule copy, movement date equal to scheduled date; 141 rows) match
no observable clock within 60 s: previous-day same flight number (its AOBT_3, schedule, EOBT_1, movement),
previous departure from the same stand, last arrival at the same stand, own schedule or network arrival
shifted by a day. Any anchor covers 4.3 % of those rows. Every fallback extreme has no network record at
all. The three largest labels (24.3, 24.3, 23.4 h) are block times about one day before the movement with
a plausible clock time. No rule or feature follows. Result: `census_2025.json`.

## 2. Neighbour sequence model, main lane

`neighbour_sequence_screen.py`. For each main-lane departure the model reads, in time order, the K = 24
nearest movements at the same airport within +/-90 minutes (arrivals and departures, both directions in
time), with 16 observable fields per neighbour: relative movement time, phase, the neighbour's own network
gap (MVT-AOBT_3 for departures; ARVT_3-MVT and BLOCK-MVT for arrivals, whose block times the ranking file
supplies), schedule delay, stand, runway, operator and type equality, flight-ID offset, and the arrival
schedule-copy flag. Two 64-dimensional pre-norm attention blocks with a CLS token pool the sequence; the
64-dimensional summary is concatenated to the tabular input of the `neural_model.py` net (embeddings,
BatchNorm, 1024-512-256 GELU, dropout 0.15, 12 epochs, batch 8,192, AdamW 2e-3 OneCycle, clip 2). FLAT is
the identical net without the branch, same seeds. No departure block time or label is read for any row;
tensors are built separately per source split from its own raw files (12 s per split).

Pre-registered rule (in the script docstring before the first fit): advance only if the two-seed SEQ mean
beats the two-seed FLAT mean by at least 0.3 s on January/July, January and July both improve, the MSE gain
survives removing the 100 most favourable rows, and the fixed blend 0.75 LGBM + 0.25 net gains at least
0.15 s more with SEQ than with FLAT. No weight or hyperparameter was chosen on the held-out months.

### January/July screen (exploratory split)

| Model | RMSE (s) | January | July |
|---|---:|---:|---:|
| Single LightGBM, 91 columns (reference) | 223.698 | 201.93 | 239.99 |
| FLAT net, mean of seeds 13 and 21 | 226.519 | 207.43 | 240.97 |
| SEQ net, mean of seeds 13 and 21 | **221.753** | 203.22 | 235.79 |
| 0.75 LGBM + 0.25 FLAT | 220.813 | 199.89 | 236.50 |
| 0.75 LGBM + 0.25 SEQ | **219.209** | 198.55 | 234.71 |

SEQ minus FLAT: 4.766 s; MSE gain 2,136.5, and 1,483.9 without the 100 most favourable rows. All ten
airports improve, from 1.23 s (EHAM) to 13.20 s (LTFM). Both the baseline-worst 1 % and the remaining
99 % improve. Blend extra with SEQ: 1.604 s. Single seeds: FLAT 229.73 / 229.59, SEQ 224.55 / 225.05.
Cost: 15 minutes per SEQ seed against 82 s per FLAT seed on this machine.

### Six-pair confirmation and permutation control

Same recipe, unchanged, on the other five source-month pairs (`run_seq_confirmation.sh`), pooled row-weighted
by `summarize_seq.py` over **2,062,577 held-out main-lane flights**. Full record: `results/seq_six_pair.json`,
per pair `results/seq_screen_result_<pair>.json`.

| Pooled, six pairs | RMSE (s) |
|---|---:|
| Single LightGBM, 91 columns (early stopping on the fold; research_250 recorded 206.694 with a fixed 1,400-tree budget) | 206.662 |
| FLAT net, two-seed mean | 207.646 |
| SEQ net, two-seed mean | **203.832** |
| 0.75 LGBM + 0.25 FLAT | 203.934 |
| 0.75 LGBM + 0.25 SEQ | **202.444** |

| Pair | 1_7 | 2_8 | 3_9 | 4_10 | 5_11 | 6_12 |
|---|---:|---:|---:|---:|---:|---:|
| SEQ minus FLAT (s) | 4.766 | 4.305 | 4.058 | 4.397 | 3.042 | 2.257 |
| Blend extra with SEQ (s) | 1.60 | 1.70 | 1.54 | 1.53 | 1.26 | 1.29 |

Pooled SEQ minus FLAT: **3.814 s**, MSE gain 1,569.3 (1,346.7 without the 100 most favourable rows). Six of six
pairs and all twelve calendar months gain (June least, 1.46 s; February most, 5.99 s). All ten airports gain:
EHAM 1.00, LSZH 1.68, EDDF 1.79, LEMD 2.04, LIRF 2.41, EDDM 2.59, EGLL 4.01, LEBL 4.11, LFPG 4.67, LTFM 9.87 s.
Tail split, baseline-defined: the worst 1 % of flat-net rows improve by 95,034 MSE inside that subset and the
other 99 % by 625; in pooled terms roughly 60 % of the MSE gain sits in the worst 1 % and 40 % in the bulk, so
the gain is tail-heavy but not tail-only. Blend extra with SEQ pooled: **1.490 s** against the single LightGBM
206.662, positive in every pair. All five pre-registered six-pair checks pass (`rule.pass` in the record).
The per-pair `advance` flag inside each pair file is only the January/July month check and is false by
construction outside pair 1_7.

Permutation control (pair 1_7, seed 13, neighbour tensors shuffled among departures within airport,
separately per split): SEQ with shuffled neighbours 229.701 s against FLAT 229.723 s on the same seed (gain 0.022 s, MSE gain 10.2 and -638.0 without the 100 most favourable rows; January -0.150, July +0.143; blend extra -0.002 s), while the unshuffled SEQ seed 13 scored 224.549 s. The whole gain is the ordered neighbour information, not model capacity. Record: `results/seq_permutation_control_1_7.json`.

## Reproduction (repository root, one fit at a time)

```bash
python3 -u -B research_250/fallback250_cv.py --project . --output .cache/rmse260 --pairs '1,7' --variants ''
python3 -u -B research_260/neighbour_sequence_screen.py --project . \
   --core-train .cache/rmse260/core_<ten-month>.parquet --core-valid .cache/rmse260/core_<two-month>.parquet \
   --output .cache/rmse260/seq-screen
research_260/run_seq_confirmation.sh .cache/rmse260/logs        # remaining five pairs + permutation control
python3 -B research_260/summarize_seq.py --screen-dir .cache/rmse260/seq-screen --output research_260/results/seq_six_pair.json
python3 -B research_260/extreme_anchor_census.py research_260/census_2025.json
```

Saved held-out predictions with `MVT_ID_mvt`, `y`, `ersatz_s`, month, airport and every model column:
`.cache/rmse260/seq-screen/seq_screen_predictions_<pair>.parquet` (not committed; on this Mac under
`/Volumes/NL360 Storage 1/media/skenderbeu/2026-09-13_research/.cache/rmse260/`).

## Limitations

- Component screen against a single LightGBM and the flat net, not the full v46 ensemble. The ensemble
  already carries five flat-net seeds at weight 0.25; replacing them with sequence nets is the integration
  question, and its answer needs the paired full-ensemble comparison on the owner's Mac (the handoff names a
  partial refit against the saved CatBoost checkpoints as the cheap route).
- January/July informed the design decision; the other five pairs are the confirmation.
- Ranking-time availability: every field exists in `ranking.parquet` for both phases; a schema preflight on
  the 2026 ranking rows (like `research_250/ranking_schema_preflight.py`) has not been run here.
- Early stopping of the LightGBM reference observes the fold, as in every research_250 screen.
