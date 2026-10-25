# Pre-registration: ordered stop/move episodes on top of v56's partial-track correction

Written 27 September 2026 on branch `claude/stop-move-episodes`, **before any model with episode features was fitted,
before any label on the screen or confirmation dates was read, and before the confirmation dates were downloaded**.
Private research: no upload, no merge. v56 and its predictions are unchanged. Codex owns the three withheld cells.

## 0. Baseline: v56, rebuilt faithfully

v56 (261.2616, commit 9884150 on `codex/v54-open-data-candidate`) corrects the pre-NM comparator on `partial_ground`
main-lane rows with three frozen LightGBM residual models over 35 columns (23 two-hour trajectory summaries, six
context fields, airport and July indicators), deployed in seven airport/month cells: EDDF, EGLL January; EHAM July;
LEBL, LSZH both months. Codex's private artefacts are unreadable here (macOS protects `~/Documents`), so his exact
`multiairport_screen.main()` was re-run from a detached read-only checkout of 9884150 on hash-identical inputs
(25 of 25 source hashes). **The three real-seed models are byte-identical to his**, and every arm's and seed's metric
equals his record (`research_320/rebuild_v56_partial.py`, `results/v56_rebuild.json`; one permutation-control pickle
differs in bytes, not in metrics). This rebuild is the v56-faithful baseline below; v45 alone is never the comparator.

## 1. Development audit (descriptive; its dates are now spent for evaluation)

`research_320/audit_v56_errors.py`, `results/audit_v56_dev.json`: v56's development test dates (January 4, 13, 20, 28
and July 9, 11, 16, 27, 2025; v56's models out of sample), the seven deployed cells, `partial_ground` rows, without the
325 rows the guarded Amsterdam detector owns in EHAM July. 9,467 rows: comparator 199.24 s, v56 186.22 s. The top 100
rows carry 39.9 % of v56's remaining squared error; LEBL July alone 37.7 %. v56's mean residual (label minus
prediction) rises with the ordered episode pattern that its 23 summaries do not encode:

| episode pattern | rows | mean residual |
|---|---|---|
| 0 / 1 / 2 / 3+ stops of 30 s or more | 67 / 24 / 6 / 2 % | -5.8 / +9.6 / +45.4 / +114.2 s |
| stopped after leaving the stand: 0, under 5, 5-15, over 15 min | 5 / 21 / 4 / 1 % | -18.8 / +19.3 / +76.4 / +193.2 s |
| last stop at the stand / more than 500 m away | 6 / 17 % | -16.2 / +53.3 s |

v56 under-predicts taxi time for aircraft that stop again after the push-back, and more so the longer and the
farther out they stop. The feature set was written before this audit and is not changed by it.

## 2. Hypothesis and arms

**Hypothesis:** the eleven ordered stop/move episode features of `research_320/episodes.py` (`EPISODE_FEATURES`:
stop count; duration, end age and stand distance of the last and the previous stop; duration and end age of the
longest stop; the last stop within 80 m of the stand; total stop time after first leaving 80 m), computed on exactly
the points behind v56's 23 summaries, reduce v56's error on its eligible rows in the seven deployed cells.

Every arm uses v56's recipe unchanged: its eight training dates (January 8, 9, 24, 30; July 2, 4, 18, 23, 2025), its
training rows (all `partial_ground` rows at the five airports, both months), target label minus comparator,
LightGBM with 200 trees, learning rate 0.03, 7 leaves, min leaf 80, L2 100, row fraction 0.8, column fraction 0.9,
max_bin 63, four threads, seeds 177, 277, 377; corrections averaged over the seeds, shrink 0.5, floor 300.

- **v56**: the rebuilt real-seed models, 35 columns. The comparator of every rule.
- **episodes**: 46 columns, v56's 35 plus the eleven episode features.
- **control 1 to 5**: 46 columns, the eleven episode columns permuted jointly within airport and date, separately in
  training and evaluation rows (seeds 3201 to 3205). Same width, row link broken.

The models are fitted once, before the screen, and reused unchanged for the confirmation (hashes recorded).

## 3. Evaluation cohort and dates

`partial_ground` rows in the seven deployed cells, without the rows the guarded Amsterdam detector estimates in EHAM
July (v56 keeps the detector there), so NM, Munich, Amsterdam-detector and fallback predictions are never touched.
Rows are the rows v56 corrects, from Codex's own extraction code run from the frozen checkout.

- **Screen**: v56's replication dates, **January 14, 21, 26 and July 13, 14, 25, 2025**. Codex used them only to
  replicate v56's frozen models; neither the audit nor the feature design read them.
- **Confirmation**: fresh dates no ADS-B study has touched, drawn with `numpy.random.default_rng(20260927)`, one
  permutation over the eligible January days, then one over the eligible July days, first four each:
  **January 2, 18, 6, 1 and July 12, 31, 26, 7, 2025**. A date that cannot be fetched completely from its preferred
  release is replaced by the next of its month's permutation (January 22, 10, 12, ...; July 6, 29, 28, ...).

## 4. Rules, identical for the screen and the confirmation

Gain per row = squared error of v56 minus squared error of the arm. The episodes arm passes a stage only if all hold:

1. pooled gain above zero;
2. above each of the five same-width controls' pooled gains;
3. January and July each above zero;
4. above zero after removing its 100 most favourable rows;
5. airport-day block bootstrap (2,000 draws) 95 % interval above zero;
6. each seed above zero (episodes seed s against v56 seed s, each alone with shrink 0.5 and floor 300);
7. at least five of the seven cells above zero.

Reported, not judged: per cell and date, per control and seed, and the screen without days 6, 13 and 20 (v56's
conservative NM-date mask). **The hypothesis holds only if both stages pass.** Then a candidate would need its own
build checks and the owner's decision; nothing is built or uploaded here. If either fails, the models stay frozen and
nothing is re-tuned on these dates.

## Addendum A, 27 September 2026: the screen fails, the hypothesis is rejected

`research_320/episode_experiment.py screen`, 10:08 to 10:12 local, under the shared lock, `results/episode_screen.json`.
The eighteen models were fitted once on v56's 13,844 training rows (the 35 v56 columns reproduce v56's training rows
exactly). **Parity:** the extraction plus the rebuilt v56 models reproduce Codex's recorded replication exactly: 15,909
rows, 9,826 changed, RMSE 185.66812426781576 to 177.09363856021687. Cohort: 6,562 rows (259 Amsterdam-detector rows
excluded).

| | episodes against v56 |
|---|---|
| pooled | **-8.8 MSE per row** (v56 175.919 s, episodes 175.944 s) |
| January / July | -70.5 / +53.3 |
| without the best 100 rows | -157.7 |
| airport-day 95 % | [-78.0, +53.5] |
| seeds 177 / 277 / 377 | +13.4 / -16.9 / -22.4 |
| cells positive | 3 of 7 (EHAM-07 +165, LSZH-07 +104, EGLL-01 +45; LSZH-01 -236, LEBL-07 -129) |
| controls 3201 to 3205 | -16.3, -11.7, -16.3, -13.2, -19.3 |
| outside days 6, 13, 20 (descriptive) | -21.7; -188.6 without the best 100 |

**Verdict: fail**, six of seven rules (only "beats all controls" holds). The hypothesis is rejected: on v56's own rows
the ordered episodes add nothing the 23 summaries do not already give the model. The audit's pattern is real, but it
is not predictive beyond what v56 already uses. No re-tuning. The confirmation stage is not run, so the eight fresh
dates stay unread: they are fetched and extracted as a reserve for a future pre-registered test.
