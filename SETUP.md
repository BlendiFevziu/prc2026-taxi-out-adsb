# Setup and reproduction

## 1. Requirements

Reference machine: MacBook Pro with M3 Pro, 18 GiB RAM, 11 CPU cores, macOS 15. Linux works too;
the neural network uses the CPU unless a CUDA GPU is present. Memory is the limit: the feature
table has about 2.1 million rows and needs about 6 GB including intermediates, so run one model
fit at a time.

Recorded versions: Python 3.9.6, numpy 2.0.2, pandas 2.3.3, pyarrow 21.0.0, lightgbm 4.6.0,
catboost 1.2.10, torch 2.8.0, joblib 1.5.3.

    python3 -m pip install --user -r requirements.txt

Run every command from the repository root.

## 2. Data and geometry

Put the fourteen competition files into `data/` (see `data/README.md`) and check them:

    python3 check_dataset.py

`geometry/` is included. `python3 fetch_geometry.py` refreshes it through the Overpass API, which
takes a few minutes; without geometry the distance features stay empty.

## 3. v44 recipe (base model, required by the v45 release)

    PRC_MODEL_OUT=models_fid_arrivals.pkl python3 main_model.py --alle
    PRC_MODELLE=models_fid_arrivals.pkl python3 submission.py

The first command trains the complete ensemble on all twelve months: about 4.5 hours on the
reference machine, most of it CatBoost, and a model file of about 1.7 GiB. Without `--alle`,
`main_model.py` trains on ten months and reports January and July as a holdout. The second command
writes a checked submission file into `submissions/` (row count, columns, key order, missing and
negative values, dtype); nothing is uploaded.

## 4. v45 release (91-column schema, full ensemble)

The command used for the team's run, with the v44 outputs from section 3:

    python3 -u -B research_250/release_pipeline.py --project . --run .cache/clock_queue_release_20260912 \
        --legacy-model models_fid_arrivals.pkl --legacy-submission submissions/upbeat-goblin_v44.parquet --jobs 8 --detach

The stages run in fresh processes with restartable checkpoints in the run folder: a replay of the
v44 model, the LightGBM length chosen on an inner split (March/September), complete baseline and
candidate ensembles trained without January/July, the paired January/July gate, the all-year final
fit, and the export to `<run>/candidate.parquet` with format checks. It never uploads. On the
reference machine the whole run took about a day.

## 5. v46 fallback anchor

    python3 research_240/anchor_preview_2026.py --project . --output .cache/research_240/anchor_preview.json
    python3 research_240/anchor_preview_2026_production.py --project . --output .cache/research_240/anchor_preview_2026_production.json
    python3 research_240/build_v46_candidate.py --output .cache/research_240/v46_rebuild/candidate.parquet

The first script builds and caches the twelve-month and ranking fallback tables; its own check uses
the research encoding and is expected to stop it after the tables are written. The second refits the
fallback with production encoding, must reproduce the v45 fallback predictions, and writes the
anchored 2026 predictions. The third replaces the 5,290 fallback predictions of the v45 file with them
and compares the result with the uploaded file's recorded hash. The scripts default to the team's
release folder (`.cache/clock_queue_release_20260912/`) and check hashes from `results/*_manifest.json`;
a new run on other hardware needs its own paths and will produce different hashes.

## 6. Research screens

The fallback screens in `research_240/` build on each other: `fallback_anchor_screen.py`, then
`fallback_anchor_production.py`, then `fallback_rescreen_production.py`, whose cached six-pair tables
the later screens (`fallback_weather_screen.py`, `fallback_blend_copy_screen.py`) reuse. The weather
screen needs `research_240/fetch_metar.py` first. Each script's docstring states its inputs, its
pre-registered decision rule and its arguments; aggregate results are in `research_240/results/` and
`research_250/results/`.

## 6b. Final phase (January, February, June, July 2026)

The final dataset adds `final_ranking.parquet` and `final_submitting.parquet` (670,790 departures).
Put them into `data/` next to the other files (`mc cp prc26/prc-2026-datasets/final_ranking.parquet
prc26/prc-2026-datasets/final_submitting.parquet data/`). The design is unchanged, only the input file
differs.

1. Base models. Sections 3 to 5, then the neighbour sequence networks of v47
   (`research_260/sequence_release_v47.py`, stages `release-prepare` and `release-fit`, five seeds,
   Apple-silicon GPU in the team's run). The copy gate is part of `research_800_final/four_month_base.py`
   (`rome_copy_gate`, from the cached fallback tables of `research_240/`). These produce the model files and
   caches that the next step reads.
2. Replication gates on the first-phase ranking file. They must pass before the final file is trusted;
   they re-create the scored v47 main lane, the v46 fallback, the copy gate and the v69/v70 Rome rules:

       python3 -u -B research_800_final/four_month_base.py --gates

3. The base for every row of the final file:

       python3 -u -B research_800_final/four_month_base.py --final

4. February and June checks and layers (optional, need aircraft tracks):

       python3 -u -B adsb-lol/fetch_months.py                       # about 190 GB of downloads, hours
       python3 -u -B research_800_final/comparator_feb_jun_2025.py  # out-of-sample comparator, February and June 2025
       python3 -u -B research_800_final/detector_feb_jun.py --check
       python3 -u -B research_800_final/detector_feb_jun.py --apply --airports <those that passed>

5. Assembly and format checks (row set and order of `final_submitting.parquet`, dtype, finiteness):

       python3 -u -B research_800_final/assemble_final.py [--layers .cache/research_800/layer_detectors_feb_jun_2026.parquet]

   January and July rows come from the scored v80 file (its ADS-B layers need the January/July tracks
   and the chain `research_360_integration` ... `research_640_tail_rule`, see `README.md`), moved by the
   known shift of `MVT_TIME_UTC_mvt`. Nothing here uploads; the file is copied to the team bucket with the
   MinIO client as `<team>_final.parquet`.

## 7. Tests

    python3 -B -m unittest discover -s tests -v

## 8. Two traps

**OpenMP.** torch and LightGBM each ship their own `libomp`. If LightGBM or CatBoost loads it first,
the neural network's first training step dies with `Segmentation fault: 11` and no error message. That
is why `neural_model` is imported before the tree libraries and why the network trains first.

**Exit codes.** `python3 script.py | tee log` hides a crash, because the exit status comes from `tee`.

## 9. Glossary for the German identifiers

    lern / pruef      training / holdout        merkmale     features
    rest              residual                  ersatz       substitute value (first guess)
    versp             delay                     hoehe        height (of an extreme)
    saat / saaten     seed / seeds              anteil       weight, share
    rueckfall         fallback                  erfunden     fabricated (copied block time)
    kappung           clipping                  halteprobe   holdout
    abgabe            submission                auslast      traffic load count
