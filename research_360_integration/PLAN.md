# Integrating a stronger base under the v62 pipeline

Prepared 29 September 2026 by Claude Code while Codex runs the pre-registered base-capacity study on
`codex/v61-base-capacity` (read with `git show`, worktree untouched). Release reference: **v62, 258.5995, NM-free**.
Lightweight only: code, tests, documentation. No fit, download, upload or merge. Complements Codex's
`research_350_base/ADSB_REVALIDATION.md`, which fixes the principles; this folder is the machinery.

## 1. How v62 is assembled: one owner per scored row

`harness.ownership_v62()` (Codex's provenance for the NM-free reference, plus the NM-free build's rows):

| owner | rows | value | depends on the base? |
|---|---:|---|---|
| base | 234,650 | pre-NM `.35/.30` ensemble (gate, 300 s floor inside) | **is** the base |
| partial | 72,977 | `max(base + 1.0 × C1, 300)` | yes: input, target, application |
| stand | 9,749 | `max(base + 0.5 × stand model, 300)` | yes: input, target, application |
| eham_january | 9,752 | `max(base + 1.0 × EHAM add-on, 300)` | yes: input, target, application |
| direct_detector | 12,423 | MVT minus the detected off-block (Munich, Amsterdam) | no, frozen |
| fallback | 5,290 | anchored fallback plus Rome gate | no, frozen |

Owners are disjoint, so application order only documents lineage: base, then partial (v56, v58, v60, v62), stand
(v57), Amsterdam January (v59), detector (v54/v55) and fallback as frozen owners. The 8,505 rows of 6/13/20 January
and July added in v61 belong to their layers. A new base must not reintroduce the old NM-day exclusion.

## 2. What each base-dependent layer reads (`layers.LAYERS`)

| layer | rows it owns (eligibility, base-free) | inputs | target | recipe | training rows (fixed) |
|---|---|---|---|---|---|
| partial | `partial_ground`, cells EDDF/01, EGLL/01, EHAM/07, LEBL/01, LEBL/07, LSZH/01, LSZH/07, not detector rows | 35: **comparator**, ersatz_s, schedule, gate, hour sin/cos, 23 track summaries, 5 airport flags, is_july | y − comparator | 800 trees × 31 leaves, seeds 177/277/377, weight 1.0 | E: v56's 8 days + Jan 16, 23, Jul 3, 21, 22, 29 (24,093 rows) |
| stand | `observed_at_assigned_stand`, cells EDDF/01, EDDF/07, EGLL/01, LEBL/01, LEBL/07 | the same 35 plus the guarded detector's taxi time (36) | y − comparator | 200 × 7, weight 0.5 | S: v56's 8 days (2,619 rows) |
| eham_january | `partial_ground` at EHAM in January, not detector rows | **comparator**, ersatz_s, schedule, gate, hour sin/cos, 7 sampling-aware track features, is_july (14) | y − comparator | 200 × 7, weight 1.0 | Codex's TRAIN dates, EHAM rows |

All recipes: LightGBM regression, learning rate 0.03, min leaf 80, L2 100, row 0.8, column 0.9, max_bin 63. The
comparator is an input **and** the target's reference **and** the application point, so a new base changes all
three. `gate` is part of the base as well; if a new base changes the reporting classifier, substitute it too
(`layers.BASE_DEPENDENT_COLUMNS`). Eligibility and cells come from the ADS-B extraction and the track-feature
drift gates, which do not read the base, so the owner lists stay frozen.

**Arithmetic.** The layers' `comparator` column was produced by `research_310/supporting_evidence.comparator`
(9884150). Codex's `metrics.compose` computes the same blend in a different order and differs by up to about
2e-13 s on some rows. Harmless for scores, but any exact-equality check (for example "row still at the comparator")
must use `layers.compose_base`, which reproduces the layers' arithmetic bit for bit.

## 3. The harness (`harness.py`) and what is proven

- `assemble_2026(base, own, models, frozen)`: the full 2026 file from any main-lane base; frozen v62 models unless
  refitted ones are passed; detector and fallback frozen.
- `policy_2025(base, rows, layer_frames, models, set_aside)`: the complete held-out main-lane policy for any base,
  all ten airports, layers re-applied on the substituted base, untouched rows at the base.
- `compare(rows, before, after, fallback)`: paired comparison with Codex's definitions (airport-day blocks within
  source month, 5,000 draws, seed 350, 95 % and 98.75 %; best-100 removal pooled and per month; per airport;
  frozen fallback scored separately and in the combined RMSE).
- `layers.fit(layer, train)`: the frozen recipe on residuals of whatever base the frame holds (not run yet).

`reproduce.py` (results in `results/reproduce.json`), unchanged base substituted everywhere:
**R1** the rebuilt 2026 file is byte-identical to v62 (0 rows differ); **R2** the recomposed 2025 base equals the
comparator of every extraction table and gives 214.6898 s, Codex's preflight value; **R3** C1 against arm 4 on
the 16,029 B1 rows replays capacity2's +1,058.98 exactly; **R4** the stand model replays stand_development cell
by cell. Reference point for any new base (old base alone → old base plus v62 layers, main lane, all airports):
B1 218.46 → 212.24 s (76,360 rows), B2 221.45 → 214.24 s (43,742 rows); with the frozen fallback 259.21 → 254.06
and 247.86 → 241.52. 14 unit tests in `tests/test_integration_harness.py`; the whole suite, 190 tests, passes.

## 4. Evaluation once Codex's study reports

1. **Receive** the new held-out January/July predictions (the refitted family's column for all 339,046 main rows,
   or the composed base). Recompose with `compose_base(rows, trees=... or cat=...)`; check IDs, finiteness and
   that his training months exclude January and July.
2. **Stale-layer diagnostic** (allowed, not decisive): `policy_2025(new_base, rows, frames)` with frozen layers.
3. **Refit the three layers** on residuals of the new base, same training rows, recipes and seeds
   (`layers.fit`). The new base's January/July predictions are out-of-training for both the layers' training and
   evaluation days, so they can feed both. Never use an all-year base (trained on these labels) here.
4. **Compare complete policies on identical rows:** old base + v62 layers against new base + refitted layers, on
   B1 (development) and B2 (replication), all ten airports, frozen fallback; report base-only and incremental
   layer effects separately, January, July, each airport, best-100, 95 % and 98.75 % intervals, each seed.
5. **Independent confirmation** on seven fresh days (below), then the 2026 assembly from the all-year retrained
   base (Codex's final training) with `assemble_2026(new_base_2026, models=refitted)`, the same byte checks as R1
   outside the changed rows, and a fixed prediction range. No upload without the owner.

## 5. Dates already inspected (January/July 2025)

| use | January | July |
|---|---|---|
| layer training S (v56, stand, EHAM TRAIN) | 8, 9, 24, 30 | 2, 4, 18, 23 |
| extra partial training E (C1) | 16, 23 | 3, 21, 22, 29 |
| B1 development (v56 test+replication; stand, capacity studies) | 4, 13, 14, 20, 21, 26, 28 | 9, 11, 13, 14, 16, 25, 27 |
| B2 replication (stand confirm, both capacity studies, stand capacity) | 1, 2, 6, 18 | 7, 12, 26, 31 |
| s = 1.0 confirmation | 5, 7, 19, 23 | 15, 17, 21, 29 |
| Amsterdam January: Codex screen / confirmation; weight test | 4, 28 / 14, 21, 26; 1, 2, 5, 7, 18, 19, 23 | none |
| detector tests (tow guard; geometry; matching) | 4, 14, 20, 25; 9, 13, 28, 30; 8, 16, 24, 26 | 3, 14, 22, 27; 2, 9, 16, 23; 4, 11, 18, 25 |
| base-model tests (Colab sets A and B) | 3, 7, 11, 15, 19, 23, 27; 5, 9, 13, 17, 21, 25, 29 | the same days |
| Codex's base-capacity screen | every day (held-out months) | every day |

Codex's NM research also read most January and early-July labels. **Every locally held January/July day has been
used at the layer level, and every January/July day at the base level.** An untouched block cannot come from local
data.

**Independent confirmation proposed:** January 3, 11, 12, 27 and July 6, 8, 10 (seven days, not held locally,
at least two days from every E training day). They were never used at the layer level; January 3, 11 and 27 were
base-only test days in Colab set A, and all seven are in Codex's base screen, which must be disclosed. Downloading
their adsb.lol tracks (about 3 GB each) and running the extraction waits until the machine is free.

## 6. Open before a new-base comparison can be complete

**Update 29 September: the detector owners and the Amsterdam January features are done (`PART1_RESULT.md`).**

- **Detector rows in 2025.** Direct-detector rows are base-free and must be set aside identically in both arms.
  Amsterdam's guarded estimates (`research_300/tow_guard_test.estimates`) and Munich's (research_290) need a
  track replay for the evaluation days; until then R5 counts them at the base in both arms, which dilutes but does
  not bias a paired comparison.
- **Amsterdam January in 2025.** Codex's features exist for 4, 14, 21, 26, 28 January (his cache) and 1, 2, 18
  (weight test); 6, 13 and 20 were NM days and were never extracted. Apply the layer on B1/B2 January only after
  that extraction.
- **2026 base.** A new base needs its all-year model's predictions for the 2026 rows before `assemble_2026` can
  run with it; Codex's study produces held-out 2025 predictions only.
