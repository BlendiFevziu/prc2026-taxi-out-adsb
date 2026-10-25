# Pre-registration: Codex's parked EHAM January add-on at full weight (s = 1.0), on unread January days

Written 28 September 2026 on branch `claude/stop-move-episodes`, **before any label of the dates below was read for
EHAM**. Owner: "test the EHAM add-on at full weight". The add-on is Codex's (`research_310/EHAM_RECOVERY_*`, commit
8d98953): seven sampling-aware track features, six context fields and a July flag, three frozen LightGBM residual models
(v56's recipe, EHAM training rows only), fixed shrink 0.5, parked because its planned gain (0.09 to 0.19 s) missed his
one-second bar. His 21 models are used as copied from his cache (hash-identical to his record); his feature extraction
is run from a read-only checkout of 8d98953. Nothing is refitted. No upload.

## Exploration (his screen and confirmation dates, spent)

`eham_weight_headroom.py`: eligible rows (partial_ground, not the guarded Amsterdam detector's), gain against the
comparator per eligible row at s = 0.5 / 1.0 / best: screen (872 rows) +4,672 / +6,951 / +7,204 at 1.25; confirmation
(1,304 rows) +3,722 / +5,226 / 1.0 best. The same under-weighting v56's own model showed (confirmed, v58).

## Fresh dates

**January 1, 2, 5, 7, 18, 19 and 23, 2025**: the eight reserve and shrink-confirmation January days without January 6
(Codex's add-on excludes days 6, 13 and 20). Those studies read labels at other airports and cells only; no EHAM
January row of these days was ever read. Rows, categories and the 23 v56 summaries come from the research_320 extraction
with Codex's frozen v56 code; his `eham_recovery.extract` recomputes and asserts them, then adds the seven features.
Eligible rows: partial_ground and not estimated by the guarded Amsterdam detector (replayed, `tow_guard_test.estimates`);
the detector's rows keep its value in every arm.

## Comparisons, both required

1. **Weight**: the add-on at s = 1.0 against s = 0.5. Controls: his five permutation-control model sets (3101 to 3105,
   his own permutation of the seven features within airport and date), scaled the same way.
2. **Value**: the add-on at s = 1.0 against no add-on (v58's policy on these rows: the comparator). Controls: his
   context arm and five permutation arms, each at s = 1.0.

Rules, applied to each comparison on the eligible rows: pooled gain above zero; above every control; positive after
removing the 100 most favourable rows; airport-day block bootstrap (2,000 draws) 95 % interval above zero; each seed
positive on its own; at least five of the seven days positive.
**Minimum meaningful**: comparison 2's gain per eligible row times Codex's 8,968 eligible 2026 rows, over 344,841,
at least **0.1 s** at full transfer (smaller than the earlier 0.3 s bars because the cell is small; stated now).

If both pass: candidate v59 = the scored v58 with the add-on at s = 1.0 on its 8,968 eligible 2026 EHAM January rows,
every other row byte-identical to v58, verified on the same seven points before the owner decides. If either fails,
nothing is built.

## Addendum A, 28 September 2026: both comparisons pass

`eham_weight_experiment.py`, `results/eham_weight.json`. Codex's extraction replayed every partial row's category and
23 summaries exactly on all seven days. 3,045 eligible rows (detector rows preserved).
**Weight, s = 1.0 against 0.5: pass, all six rules.** +2,312 MSE per row, +477 without the best 100, airport-day
[1,442, 3,548], seven of seven days positive, seeds +2,320 / +2,308 / +2,289; his five controls scaled the same way lose
867 to 1,088.
**Value, the add-on at s = 1.0 against no add-on: pass, all six rules.** +6,980 MSE per row, +2,898 without the best
100, airport-day [4,894, 9,882], seven of seven days, seeds +6,921 to +6,999; context and the five controls at s = 1.0
lose 1,263 to 1,655.
Planning on Codex's 8,968 eligible 2026 rows: 181.5 MSE per scored row, **+0.35 s at full and +0.17 s at half
transfer** against v58; the 0.1 s minimum holds. Next as registered: v59 on the scored v58, verified before the owner
decides.

## Addendum B, 28 September 2026: the official result

After the seven-point verification the owner authorised the upload ("upload v59"). `upload_v59_once.py` uploaded the
candidate once at 06:55:04 UTC as `upbeat-goblin_v59.parquet` (4,727,754 bytes, remote equal to local). **Official
score 259.4958** against v58's 259.9035: a gain of **0.4077 s**, slightly better than the full-transfer end of the range
fixed before the upload (259.55); 211.8 MSE realised against 181.5 predicted. Record: `results/upbeat-goblin_v59.json`.
