# Sequence airport follow-up, 14 September 2026

Research branch: `codex/sequence-airport-ablation`, starting from the team's 7095116.
The owner authorized this comparison. No upload or merge to main is authorized.

## Why this is a new experiment

the team's sequence builder uses ADEP for both phases. Arrivals occur at ADES. In
January/July, 99.2% of the arrival neighbours it selects actually land elsewhere.
His component gains remain useful evidence, but do not establish that local
arrival congestion is responsible. Keep his original origin-context model as a
control; compare a separate model grouped by the actual movement airport.

`sequence_features.py` implements both modes. Each still selects 12 preceding
and 12 following movements within 90 minutes. `sequence_model.py` preserves the
network recipe and saves raw residuals and trained state. Do not apply the
300-second floor to each seed before averaging in the production ensemble.

The exact pass rules were committed before interpreting candidate results in
`airport_ablation.py`. First two matched seeds, FLAT/origin/local. Then a frozen
full-ensemble comparison at the existing neural weight of 25%. Extend promising
candidates to five seeds only after the registered gates pass. Corrected local
features still require independent source-pair and permutation confirmation.
Weather and direction additions are a later experiment, not mixed into this one.

## Verified before the sequence fits

- 80 tests pass. The new tests cover airport isolation, departure-label
  independence, timestamp units and shifts, query ordering, input contracts,
  original network/preprocessing parity and floor/gate order.
- `sequence_inference.py` preserves the research treatment of unseen categories.
  Its preprocessing exactly reproduces cached inputs on all 339,046 validation
  rows; saved FLAT weights replay from raw feature rows with a maximum CPU/MPS
  difference of 0.001465 s on the first 4,096 rows (same-device replay is exact).
- The origin builder matches the team's tensors exactly on 2,055 real held-out rows.
- Both builders produce finite tensors on all 339,551 main ranking rows.
- Airport/month drift audit flags nine feature cells above 0.25 reference SD,
  mainly January schedule delay and some arrival reporting gaps. These are review
  warnings, not automatic rejection. The original pooled drift check hid them.
- FLAT seed 13 replicates the team: 229.727356 vs 229.726620, difference 0.000736 s.
- Saved full main ensemble replay: maximum difference 1.82e-12 s. Holdout main
  RMSE 216.773463; overall 306.902244 with unchanged v46 fallback on 5,373 rows.
  These are local validation scores, not the official 268.2394 leaderboard score.

Aggregate baseline/audit record: `results/sequence_airport_preflight.json`.

## Local run and reproduction

Worktree: `~/Documents/New project/skenderbeu-selector`.
Ignored output: `.cache/sequence_airport_20260914`.
Original release cache is read-only at
`.cache/clock_queue_release_20260912`.
No packages were installed. Real fits use the Mac's MPS GPU, one at a time.
The restricted sandbox does not expose MPS; do not silently fall back to CPU.

The preparation command, run once per immutable recipe, is:

```bash
python3 -u -B research_260/airport_ablation.py \
  --release /path/to/clock_queue_release_20260912 \
  --output .cache/sequence_airport_20260914 prepare
```

For each of `flat`, `origin`, `local`, and seeds 13 and 21:

```bash
python3 -u -B research_260/airport_ablation.py \
  --release /path/to/clock_queue_release_20260912 \
  --output .cache/sequence_airport_20260914 fit --mode origin --seed 13
```

Read `*_13.log` and `*_21.log` for epoch progress. A corresponding result JSON
means a fit completed and its saved-model replay passed. Check processes and
existing files before launching anything. A failed replication exits nonzero.
The currently queued run proceeds continuously, not on a 30-minute schedule.

`sequence_ensemble_check.py baseline` reconstructs the saved ensemble once, then
`compare --seeds 13,21` consumes the new predictions. Both commands require the
release path, the v46 fallback holdout parquet and output directory. Its `--help`
lists arguments. Candidate predictions, weights and detailed audit arrays remain
in the ignored cache, never in Git. The original scored weights and submission
files remain untouched.

## Status

The paired sequence fits are running. No candidate RMSE gain has been claimed.
Read the current cache results and HANDOFF before resuming; this section will be
updated when the paired results are available.
