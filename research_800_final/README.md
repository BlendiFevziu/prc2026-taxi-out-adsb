# Final phase (January, February, June, July 2026)

One file was submitted: `upbeat-goblin_final.parquet`, 670,790 departures. The design is the one of the first phase;
only the input file changed. What is here:

| file | job |
|---|---|
| `four_month_base.py` | The frozen base for any ranking file: the v45 all-year ensemble with the five neighbour sequence networks (weights 0.35 / 0.30), the v46 anchored fallback, the Rome copy gate and the v69/v70 Rome rules. `--gates` re-creates the scored v47 and v70 artefacts on the first-phase file first (all pass, see `results/replication_gates.json`), `--final` runs the final file. |
| `assemble_final.py` | Combines the sources and checks the result against `final_submitting.parquet` (row set and order, dtype, finiteness). January/July rows are the scored v80 plus the known shift of `MVT_TIME_UTC_mvt`. |
| `comparator_feb_jun_2025.py` | A single LightGBM trained without the month, to predict February and June 2025 out of sample. Weaker than the production ensemble. |
| `detector_feb_jun.py` | The two training-free stand detectors (Munich, Amsterdam). `--check` tests them on 2025 days against the comparator under the project's four decision rules; `--apply` produces the 2026 rows. |
| `label_mechanisms_by_airport.py`, `identity_scan.py` | Descriptive scans of the 2025 labels: whether any airport other than Rome writes the schedule (or a shifted schedule) into the block time on a scale that matters. It does not (`results/*.txt`). |
| `results/` | Gates, detector check, the assembly report and the scan outputs. |

Order of a run, with the commands, is in `SETUP.md` section 6b. Decisions that are easy to miss:

- **Export dither.** In `final_ranking.parquet` only `MVT_TIME_UTC_mvt` (and for arrivals `BLOCK_TIME_UTC_mvt`) differ from
  `ranking.parquet`, by up to 12 s on 92 % of the January/July rows. For a departure the label is `MVT_TIME` minus the
  hidden block time, so the first-phase prediction moves with the known shift. Checked: the detector values recomputed from
  the new file equal v80's detector values plus the shift on all 345 rows of two test days.
- **What February and June do not get:** the partial-track correction, the stand model and the Amsterdam January add-on.
  They need same-month tracks and out-of-sample base predictions for training and were not re-fitted.
- **11 June 2026** has no adsb.lol archive; its rows keep the base prediction.
