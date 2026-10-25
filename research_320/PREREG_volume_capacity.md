# Pre-registration: is v56's partial-track correction limited by training volume or by capacity?

Written 28 September 2026 on branch `claude/stop-move-episodes`, **before any fit of this experiment**. Challenge data and
adsb.lol tracks only; no NM feature, label, prediction, weight or parameter. No candidate, no upload. The current best
is v59 (259.4958); v58 and v59 carry this model at full weight in the same seven cells, so nothing here depends on
which of them is the base. Any later candidate would be built on the then-best scored file and reviewed separately.

## Inventory (done before registering)

- **No equivalent study.** Capacity was studied for the main lane (research_280/mainlane_capacity.py), the fallback lane
  (rejected) and the NM add-on (v52, 200x7 to 1500x31, passed); never for the ADS-B partial-track model, which was
  trained once on 13,844 rows of eight dates with 200 trees and 7 leaves (Codex, research_310/multiairport_screen.py).
- **Labelled track days held locally**: January 1, 2, 4-9, 13, 14, 16, 18-21, 23-26, 28, 30 and July 2-4, 7, 9, 11-18,
  21-23, 25-27, 29, 31, 2025 (42 days); 40 GB free. The residual target needs the held-out production-strength
  comparator, which exists for January and July 2025 only, so training can only grow inside those two months.
- **Date reconciliation.** Claude's studies read partial rows in the deployed cells on v56's development and replication
  days and on 5, 7, 19, 23 January and 15, 17, 21, 29 July (the s = 1.0 confirmation). Codex's NM branch cites labels on
  most January days and early July (among them January 1, 2, 6, 18 and July 7). **No locally held block is untouched for
  these rows.** The limitation is stated; a pre-declared blocked replication is used and not called fresh confirmation.

## Blocks and embargo (frozen)

- **Development block B1**: January 4, 13, 14, 20, 21, 26, 28 and July 9, 11, 13, 14, 16, 25, 27 (v56's development
  test and replication days).
- **Replication block B2**: January 1, 2, 6, 18 and July 7, 12, 26, 31 (Claude's reserve days; their deployed-cell
  partial rows were never read by Claude's studies; five of them appear in Codex's NM work).
- **Embargo**: every training day is at least two calendar days from every B1 and B2 day (one intervening day), the
  rule v56 used.
- **Small training set S**: v56's own eight days, January 8, 9, 24, 30 and July 2, 4, 18, 23 (all comply).
- **Expanded training set E**: S plus every other locally held labelled day that complies: January 16, 23 and July 3,
  21, 22, 29 (14 days, about 1.75 times the days). No download. The seven further compliant days that are not held
  locally (January 10, 11, 31; July 1, 5, 19, 20) are a possible bounded next step, not part of this registration.

## Arms (35 columns, matching, eligibility, cells, floor 300 and weight s = 1.0 unchanged; seeds 177/277/377 paired)

| arm | training | capacity |
|---|---|---|
| 1 (reference) | S | current: 200 trees, 7 leaves, min leaf 80, L2 100, learning rate 0.03, row 0.8, column 0.9, max_bin 63 |
| 2 | E | current |
| 3 | S | **larger, fixed now: 400 trees, 15 leaves**, everything else as current |
| 4 | E | larger |

Arm 1 refits the deployed recipe and must reproduce the rebuilt deployed models' predictions exactly; the deployed
models are also kept as a separate fixed reference. Descriptive only: **E-fixed**, a seeded random sample (seed 4242)
of E's rows equal in number to S's, current capacity, to separate more rows from more days.

## Evaluation and rules

Rows: `partial_ground` rows in v56's seven deployed cells, without the guarded Amsterdam detector's rows (v56's
precedence); everything else is outside the comparison and untouched. Gain per row = squared error of arm 1 minus
squared error of the arm, both at s = 1.0. Arms 2, 3 and 4 are three comparisons: every interval below is a
**Bonferroni-adjusted** airport-day block bootstrap interval (2,000 draws, 1 - 0.05/3 two-sided).

An arm **survives development** (B1) only if: pooled gain above zero; January and July each above zero; above zero
after removing the 100 most favourable rows; adjusted interval above zero; each paired seed above zero; at least five
of seven cells above zero.
For each surviving arm, **controls** under its own data and capacity: context-only (the 23 track columns zeroed) and
five joint track permutations within airport and date (seeds 3101 to 3105, training and evaluation rows separately),
all at s = 1.0; the arm's gain over arm 1 must exceed every control's gain over arm 1 on B1.
**Replication** (B2, frozen models, no refit): the same six rules for every arm that survived, adjusted for the number
of survivors. The best development arm is never promoted on its own development result.

Reported for every arm: January and July, each cell and date, every seed, untrimmed squared error, best-100 removal,
intervals, gain against the pre-ADS-B comparator as well. **2025-to-2026 support** for any passing arm: v56's release
drift rule per cell with the arm's training rows as the reference against Codex's 2026 feature table, label-free.
A coverage-adjusted scenario (2026 eligible rows per cell x B2 gain per row, over 344,841, against v59) is reported
only after the controls pass; 0.1 s at full transfer is the bar for proposing a separately reviewed candidate.
Every fit runs under the shared lock, one arm at a time, released between arms.

## Addendum A, 28 September 2026, written before any fit or extraction of this experiment

1. **Multiplicity.** The brief asks to account for four tested arms. Every development interval therefore uses
   1 - 0.05/4 (99.375 % two-sided), stricter than the three comparisons with arm 1 strictly need. Replication stays
   adjusted for the number of arms that reach it.
2. **Order and reproduction gates.** Arm 1 is fitted first. Before any B1 label is read: arm 1's corrections on the
   B1 rows must equal the rebuilt deployed models' corrections seed by seed; those deployed models must reproduce
   Codex's saved development predictions (s = 0.5) exactly; the B1 partial rows on his eight test dates must equal his
   saved rows in all 35 columns; S must have his 13,844 rows. Otherwise the run stops with no result.
3. **Parity.** January 23, July 21 and July 29 were extracted before for the s = 1.0 confirmation; the new extraction
   of E's six days must reproduce those three days' rows exactly (35 columns and label).
4. **Drift replay.** The drift rule is Codex's own function; with S as the reference it must return his seven allowed
   cells before it is applied to any passing arm.
5. **Locking.** The extraction, the Amsterdam detector replay and each arm's fits each hold the shared lock; between
   holds the lock is released for 35 s, longer than the shared helper's 30 s poll. B2 is loaded only if an arm passes
   development and its controls.

## Addendum B, 28 September 2026: results

`volume_capacity.py`, run 10:35 to 10:44 local, `results/volume_capacity.json`. Independent replay in a fresh process
(`volume_capacity_replay.py`, `results/volume_capacity_replay.json`): every saved correction and every headline number
equal. **All gates held**: S has Codex's 13,844 rows; arm 1's three models are byte-identical to the deployed ones; the
deployed models reproduce his saved predictions; the three re-extracted days equal their earlier extraction; the drift
replay returns his seven cells. E has 24,093 training rows. B1 has 16,029 eligible rows (584 guarded detector rows
excluded); against the pre-ADS-B comparator, arm 1 gains +7,597 MSE per row there.

Development, B1, gain per eligible row against arm 1 (the deployed correction), intervals at 99.375 %:

| arm | gain | without best 100 | interval | January | July | seeds 177 / 277 / 377 | cells | verdict |
|---|---|---|---|---|---|---|---|---|
| 2, E x current | +73 | -196 | [-116, 259] | -140 | +300 | +98 / +94 / +31 | 4 of 7 | fail |
| 3, S x larger | +3,008 | +2,009 | [2,345, 3,817] | +2,753 | +3,278 | +3,006 / +3,059 / +2,917 | 7 of 7 | pass |
| 4, E x larger | +3,362 | +2,268 | [2,593, 4,247] | +2,804 | +3,955 | +3,322 / +3,404 / +3,328 | 7 of 7 | pass |
| E-fixed (descriptive) | -165 | -487 | [-395, 49] | -349 | +30 | -148 / -173 / -175 | 2 of 7 | - |

Controls under each arm's own data and capacity: context and the five permutations lose 7,597 to 7,978 against arm 1
(arm 3) and 7,558 to 7,825 (arm 4). Both arms beat all of them.

Replication, B2 (8,894 rows, frozen models, 97.5 % intervals for two arms): **arm 3 passes**, +3,503, +1,642 without
the best 100, [2,855, 4,337], January +3,187, July +3,877, seeds +3,554 / +3,467 / +3,443; **arm 4 passes**, +3,940,
+1,829, [3,088, 5,051], +3,563 / +4,386, seeds +3,916 / +3,946 / +3,924. Both are positive in all seven cells and on
all eight dates, as on all fourteen B1 dates.

Drift, each arm's own training rows as the reference: all seven cells allowed for both. Scenario against v59 (2026
eligible rows per cell times the B2 gain per row, over 344,841): **arm 3 669 MSE per scored row, +1.29 s at full and
+0.65 s at half transfer; arm 4 756, +1.46 s and +0.73 s.** The 0.1 s bar holds for both.

Reading: the correction is limited by capacity, not by volume. 75 % more rows do nothing at 200 x 7 (arm 2) and add
+354 to +437 at 400 x 15 (arm 4 against arm 3, below). The deployed correction is too narrow (2026 p1 / p99 -151 /
+205 s against -233 / +291 s for arm 3), the same under-fit the s = 1.0 test exposed.

Checks after the pass, descriptive (`volume_capacity_support.py`, `results/volume_capacity_support.json`). On v56's
66,292 scored 2026 overlay rows, where the deployed models reproduce Codex's 2026 corrections exactly, the larger
models behave as on 2025: RMS change against the deployed correction 39 s (arm 3) and 46 s (arm 4) in 2026, B1 and
B2 alike; correlation with it 0.94 and 0.92 in all three; the 300 s floor binds on 0.7 to 0.9 % of rows in all three.
**Not pre-registered:** arm 4 beats arm 3 by +354 on B1 [142, 583] and +437 on B2 [151, 747] (95 %), every seed.
No candidate built. Proposed as the next, separately reviewed step: v60 = scored v59 with the frozen arm 4 (or arm 3)
models replacing the deployed correction on the same 66,292 rows.
