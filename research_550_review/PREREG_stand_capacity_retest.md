# Pre-registration: stand-model capacity, retested on the fresh block the original result asked for

Written 30 September 2026, **before any row of the block below is scored with the capacity models**. Part of the
owner's review of closed lines ("keep reviewing all experiments that you think are worth a second look"). NM-free,
no fit (the models exist), no upload without the owner.

## Why this line is reopened

`research_320/PREREG_stand_capacity.md` (28 September): trained on E (4,457 stand rows) at weight 1.0, every arm beat
the then-deployed stand model on B1 and B2 pooled (K2 400 x 15: +4,794 and +6,930 per stand row) and failed only
"without the best 100" on B2's 1,102 rows. Its closing sentence: "A fair retest would need a larger untouched block".
Since then twenty 2025 days were downloaded (29 September: fresh_july 7, eddf_july_confirm 3, fresh_january 10). Their
stand rows were scored only with the DEPLOYED stand model (weight tests of research_440/450), never with K1-K3.

## Comparison, fixed now

Reference: the deployed stand model (S, 200 x 7, `research_320/stand/models/real_{seed}`) at weight 1.0, as in v67 to
v70. Decision arm: **K2** (E, 400 x 15, `stand_capacity/models/k2_real_{seed}`) at weight 1.0, chosen because it was
closest to passing the failing rule on both earlier blocks (B1 +674, B2 -36). K1 and K3 reported, not judged.
Rows: category observed_at_assigned_stand in the stand model's nine 2026 cells (EDDF 01/07, EGLL 01, LEBL 01/07,
LSZH 01/07, EHAM 01/07) on the twenty days, Amsterdam detector rows excluded; comparator bit-identical to the pre-NM
base. Floor 300.

## Rules, all must hold for K2

1. Pooled gain per row above zero.
2. Above zero without the 100 most favourable rows.
3. January and July each above zero.
4. At least 70 % of days above zero.
5. At least two thirds of the cells with rows above zero.
6. Every paired seed above zero.
7. Planning: gain per row times 17,147 (the stand model's 2026 rows), over 344,841, against v70's central 253.57, at
   least 0.1 s.

A pass leads to v71 = v70 with K2 in place of the deployed stand model on its 17,147 rows, built and verified like v70.

## Result, 30 September 2026: K2 passes every rule; v71 built and verified

`stand_capacity_retest.py`, `results/stand_capacity_retest.json`. 5,862 stand rows over the twenty days, comparator
bit-identical. **K2: +2,672 per row, +1,288 without the best 100, January +2,873, July +2,459, 9 of 9 cells, every
seed, planning +0.26 s. Pass.** Reported: K1 +775 (fails only the 0.1 s planning bar), K3 +3,101 (LEBL July -116).
`build_v71.py`, `verify_v71.py`: all 17,147 stand rows reproduced v70 with the deployed model first; 16,948 change
(RMS 47.8 s); everything else identical to v70; the verifier recomputed both corrections through
stand_experiment.inputs with the base from the pre-NM file. SHA 35145d28... Prediction fixed in the build before it
ran: central 253.36, best 253.26, worst 253.49 (from v70's central).
