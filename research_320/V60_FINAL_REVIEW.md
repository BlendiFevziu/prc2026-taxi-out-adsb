# v60: submitted once, 258.2119 RMSE

On 28 September 2026 at 10:00 UTC the owner-authorized corrected candidate was
uploaded once as `upbeat-goblin_v60.parquet`. The official result is Succeeded,
344,841 pairs, **258.2119 RMSE**, **1.2839 s better than v59 (259.4958)**.
Remote file bytes were downloaded and SHA-256-verified against the local file.
No upload was retried and no parameter was changed from the score.

Correct candidate SHA-256:
`ed9592271c5942f681376c305fc2680e6c6d1a877e785ffc25f0fa270c663b30`.
Receipt, manifest and verification are under `v60_corrected/` in the original
Codex research folder and copied to Claude's `.cache/research_320/volume_candidate/`.
The canonical repository receipt is `results/upbeat-goblin_v60.json`.

## Final review found and fixed two assembly errors

The first local candidate (`550c4f7f...`, never uploaded) used the obsolete
67,240-row overlay and added arm4's correction on top of v57, which already
contains half of the old correction. It consequently stacked corrections and
changed 948 protected Amsterdam detector rows. The first verifier duplicated
the builder's formula; its "verified" conclusion is withdrawn. Those files
are retained for audit, and their entry points now stop with an invalid notice.

The corrected candidate starts from exact scored v59. For exactly the deployed
66,292 partial-track IDs, it replaces the prediction with
`max(original comparator + mean(arm4 seed predictions), 300)`.
It changes 65,879 rows; 278,962 rows remain identical to v59. Arm4 is the frozen
14-date, 24,093-training-row, 400-tree/15-leaf model, seeds 177/277/377, weight 1.0.
There was no fit or selection from leaderboard feedback.

## Review evidence

All input hashes and scored release receipts v51/v52/v55/v56/v57/v58/v59 checked.
The old arm replays v59 on the eligible rows exactly. The new arm replays the
candidate in reverse row order exactly. Submission schema, dtypes, finite values,
IDs and order pass. All 35 model fields retain their frozen order.

Protected groups equal v59 exactly: existing NM effect (31,160 visibly changed
v52-versus-v51 rows), all Munich rows (25,966), Amsterdam detector (4,188),
fallback (5,290), v57 stand changes (8,676) and all EHAM January add-on eligible
rows (8,968, of which 8,919 visibly changed). The 66,292 eligible rows overlap
none of these groups, and every other row remains identical. No new NM sample,
prediction, model, fitted parameter or weight enters this release. The inherited
NM component remains as explicitly authorized by the owner on 26 September.

Validation formula replayed directly on B1 and B2 with the same saved models:

| Block | Gain in MSE per eligible row | Gain excluding best 100 | January | July |
|---|---:|---:|---:|---:|
| B1 | 3361.988 | 2268.196 | 2804.109 | 3954.681 |
| B2 | 3939.507 | 1828.957 | 3563.012 | 4385.548 |

Each matches Claude's registered report. Permutation/control gates and all seven
allowed cells' drift gates pass. Previously inspected dates are disclosed;
these are blocked replication results, not untouched confirmation.

The scenario fixed before upload was **258.0358 at full transfer, 258.7668 at
half transfer**. Actual 258.2119 lies between them. The earlier candidate did
not implement the validated recipe, so these estimates never justified uploading it.

## Handoff

v60 is the current scored reference. Do not upload it again or reapply the
partial-track or EHAM components. Keep the existing NM sample frozen; future
NM experiments remain research-only. No model fit, upload or background job is
left running. Future experiments need a new hypothesis and comparisons against
v60; the capacity/volume experiment and ACARS matching pilot are already done.

Implementation: `v60_release.py` (correct assembly), `v60_review.py` (independent
recipe/lineage/validation review), `v60_upload_once.py` (intent, quota, duplicate,
hash and one-upload guards). Source copies are in `research_320/codex_v60_sources/`.
