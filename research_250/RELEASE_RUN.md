# Authorized full-model release run — 12 September 2026

The user approved full validation/retraining and **one new submission after checks
pass**. At about 21:20 CEST on 12 September they also authorized a GitHub commit/push
after this release and official-result record finish, followed by the friend's
final research prompt. Leaderboard tuning remains unauthorized.
**Completed release:** v45 scored **269.4753 seconds**, improving v44 by **2.8587**.
Training finished at 22:08 CEST, export at 22:10, and the official result was retrieved
at 22:24 on 12 September. See results/upbeat-goblin_v45.json and its manifest.
The run's upload_receipt.json records the single completed upload. Do not repeat it:
ready_for_upload.json is the immutable pre-upload export record, not upload status.
All 29 regression tests passed again before handoff. GitHub publication and the
friend prompt remain the authorized next steps. Do not edit the unrelated
`Documents/New project` workspace.

## Durable state

Repository: `.`

Run: `.cache/clock_queue_release_20260912/` (ignored by Git).
Started at 2026-09-11 23:20 UTC / 12 September 01:20 CEST. Initial controller PID
39996; use `state.json`, not that historical PID, for current process identity.

- `run.json`: frozen code/input hashes, source months, library versions and policy.
- `state.json`: controller status, stage and worker PID.
- `progress.json`, `run.log`: component progress and subprocess diagnostics.
- `<stage>/checkpoints/`: restartable neural/tree models, bound to the run identity.
- `parity.json`: replay of the actual scored v44 predictions.
- `tuning.json`: candidate training length selected using inner training data only.
- `baseline/` and `candidate/`: full paired holdout models, predictions and scores.
- `gate.json`: paired untrimmed scores, airport effects and baseline-defined tails.
- `final/model.pkl`: separate all-year candidate; never replaces scored v44.
- `candidate.parquet`, `ready_for_upload.json`: locally verified output and hashes.

The controller runs detached under `caffeinate -i` to prevent idle system sleep.
Keep the Mac powered on; this does not override lid-close sleep or shutdown.
A native follow-up in this task checks every 30 minutes, stays quiet while nothing
actionable changes, and continues the authorized upload/result workflow. Automation
ID: `finish-skenderbeu-validation-and-submission`. Keep the app running for it.

## Frozen sequence and decision

**17:23 CEST update:** the complete paired gate passed at 17:00 CEST. Overall
RMSE improved 309.577417 -> 307.264321 (2.313097 s); main improved
220.091255 -> 216.773463 (3.317791 s). Fallback predictions are identical,
all ten airports improve, and both baseline-defined tail groups improve.
See `.cache/clock_queue_release_20260912/gate.json`. Final all-year training
started automatically and is fitting its first CatBoost after completing the
five neural fits, reporting classifier and five global LightGBMs. Export and
submission remain pending. These are local validation results only.

1. Replay the scored model against ranking inputs; tolerance permits only tiny GPU
   floating-point differences (max 0.05 s, RMS 0.005 s). This checks inference, not RMSE.
2. On the ten outer-learning source files, hold March/September out internally and
   select the candidate LightGBM length (max 3,000; early stopping 100; seed 177).
   Use `ceil(best_iteration * 1.05)` in both candidate validation and final training.
3. Train a complete v44-recipe comparison on the ten source months excluding
   January/July: five LightGBM seeds, three CatBoost seeds, five neural seeds,
   ten airport models, reporting classifier and fallback. Legacy LightGBM stays 890.
4. Train the same full ensemble with the explicit 91-column corrected-clock/runway
   schema. Reuse the comparison fallback exactly; all blends, gates and floors stay
   fixed. No tuning on the outer holdout or ranking feedback.
5. Evaluate both on separately built January/July source references. Advance only
   with at least 0.3 s main RMSE gain, positive overall RMSE gain, and bit-identical
   fallback predictions. All rows, including extremes, remain in the primary metric.
6. If passed, fit the candidate on all twelve 2025 source files. Copy the final
   fallback directly from the scored v44 bundle, keeping its inputs unchanged.
7. Export against the official template, run its six format checks plus finite-value,
   schema, unique-ID and exact scored-fallback checks. Stop at `ready_for_upload`.

This is **one complete January/July paired ensemble holdout**, supported by the
previous six-pair single-LightGBM evidence. It is not six-pair full-ensemble CV or
an unbiased final performance estimate: January/July also informed earlier research.
The initial 10–13-hour estimate was superseded at 10:15 CEST: baseline validation
took 7 h 39 min and recent CatBoost seeds take about 2.5 hours each. Candidate training
is active; allow roughly 16–19 further hours at that pace if validation passes and
the all-year final fit proceeds. This is an estimate, not a deadline guarantee.

## Safe restart

Read the run state/log and verify its process identity first. Do not start concurrent
fits. The controller's lock is inherited by its worker, so an orphaned live worker
also prevents duplicate training. A completed checkpoint is reused only when its
recipe and run identity match. Frozen runtime code/input/library changes stop the
run; do not edit those files while training. Documentation/tests may be updated.

From the repository, the exact start/resume command is:

```bash
python3 -u -B research_250/release_pipeline.py --project . --run .cache/clock_queue_release_20260912 --legacy-model ../models_fid_arrivals.pkl --legacy-submission ../submissions/upbeat-goblin_v44.parquet --jobs 8 --detach
```

If a stage fails, diagnose the recorded error before resuming. Never relax a failed
validation gate. The script uses fresh processes, imports torch before tree libraries,
and trains neural components before tree fits. It never uploads by itself.

## Authorized upload and result

After `ready_for_upload`, verify the run's successful gate/parity/export records and
artifact hashes. Recheck the original scored hashes against `results/upbeat-goblin_v44_manifest.json`.
Use `/opt/homebrew/bin/mc` with the existing authenticated alias; do not print or copy
credentials. Refresh `mc ls --json prc26/prc-2026-upbeat-goblin/` and local submissions
in both repository and parent folder before selecting a fresh `upbeat-goblin_vN.parquet`
name. Do not assume v45 is still unused or overwrite any object.

Record the selected object key, local candidate SHA-256 and timestamp in a new
`upload_intent.json` inside this run **before** the upload. Upload exactly this candidate
to that one object. Record the outcome in `upload_receipt.json`; on uncertain failure,
inspect the remote object and compare its bytes/hash before considering a retry.
Never create another version merely because the first response is delayed.

Retrieve the corresponding result JSON when it appears in the bucket. Preserve the
raw official response, add a compact result/manifest under repository `results/`,
and update `STATUS.md` and `experiments.csv`. Report the official RMSE and difference
from 272.334. A rank needs a fresh leaderboard check; the old 9/111 is only a dated
snapshot. Do not derive or promise the official gain from the component improvement.

If the service rejects the attempt (including a quota rejection), report the exact
status and seek direction before another submission. Pause the native follow-up if
validation fails or user direction is required. Do not upload extra variants or tune
against the returned score. Recording the official result now advances to the GitHub
handoff below rather than ending the follow-up.

## Authorized post-release GitHub and friend handoff

After the release and official result are recorded:

1. Update STATUS, README, shared agent instructions, the experiment log and compact
   validation/result evidence. Keep local and official results distinct; retain v44
   as best if the candidate is worse.
2. Prepare FABLE_RESEARCH_PROMPT.md for the user to forward, with the exact published
   candidate branch, updated results, 240 milestone / 230 stretch / first-place goals,
   calibrated team-profile hypotheses, six-hour first research pass and one-seed
   screening permission. Include one fit at a time, the reference Mac's 18 GiB RAM /
   11 CPU cores, and torch before tree libraries on macOS. Have the friend's terminal
   agent handle compatible project-local dependency setup with normal approvals,
   not system/admin modifications or paid cloud provisioning. Request a research
   branch, experiment rows, report and local predictions; no merge or competition
   upload. Do not actually send an external message to the friend.
3. Review an explicit staging list of code, tests, documentation and compact aggregate
   results. Exclude credentials, weights, caches, private machine configuration and
   generated row-level predictions/submissions. Inspect file sizes and potential
   secrets, run lightweight tests and git diff --check.
4. Verify the remote is exactly private the private working repository and refresh refs.
   Create or safely resume an identifiable codex/ candidate branch. Preserve main,
   collaborator branches and unrelated changes; do not merge or force-push.
   Commit and push only reviewed in-scope work.
5. Verify the remote commit SHA and save a receipt in the ignored run directory.
   Return the branch link, official result and final copy-paste prompt. Keep the
   existing completion follow-up active through this handoff, then delete it and
   tell the user it stopped. Report and pause for real permission/coordination
   blockers; do not bypass safeguards.

This supersedes earlier notes forbidding a GitHub push in this round. It does not
authorize new experiments, extra submissions, public redistribution or changes to
the currently frozen release recipe.
