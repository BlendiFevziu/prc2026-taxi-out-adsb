# Pre-registration: Heathrow July and the C1 + CatBoost blend, confirmed on seven fresh July 2025 days

Written 29 September 2026 while the tracks download, **before any label of these days is read and before any fit**.
Owner's request of the same day. NM-free. No candidate is uploaded without the owner.

## Days

1, 5, 19, 20, 24, 28 and 30 July 2025: the July days not held locally until now, so never used by any layer, screen or
confirmation at the layer level (all lie in the base's held-out months; Codex's base screen read every July label at the
base level; disclosed). First half 1, 5, 19, 20; second half 24, 28, 30. Tracks from each day's preferred adsb.lol
release (`research_300/fetch_preferred.py`), rows and features from the frozen v56 extraction (`stand_experiment.extract`),
base = the held-out comparator, bit-identical to the recomposed base. Detector-owned rows are excluded as before.

## A. Heathrow July with the frozen C1 (weight 1, floor 300)

Earlier today EGLL July failed on 6/8/10 July only because those days held 233 thin rows (+23,437 pooled, -20,679 without
the best 100). Rules on the EGLL July `partial_ground` rows of the seven days, all must hold:

1. Pooled MSE gain over the base above zero.
2. Above zero after removing the 100 most favourable rows.
3. Above zero in each half.
4. Above zero within the degraded and within the well-observed group (a group under 30 rows is not judged).
5. Reweighted to EGLL July's 2026 mix (degraded 6.19 %), each stratum at its gain on these days when it has at least 30
   rows there, otherwise at C1's gain for that stratum in EGLL January (the sister month, deployed since v56;
   `research_360_integration/results/error_by_track_quality.json`), times EGLL July's base-owned 2026 partial rows over
   344,841 against v65's 256.6145: **at least 0.1 s at full transfer**.

## B. The C1 + CatBoost blend (F of `research_430_robust/`) against C1

F failed only "without the best 100 rows" on B1 and B2 (+447 and +356 pooled). The same frozen models (C1 and the three
CatBoost seeds fitted on E, `.cache/research_430/models/cat_f_E_*.pkl`), no refit. Rules on the partial rows of the four
July cells C1 corrects (EDDF, EHAM, LEBL, LSZH July; EGLL July only if A passes), all must hold:

1. Pooled gain of F over C1 above zero.
2. Above zero after removing the 100 most favourable rows.
3. Above zero in each half.
4. Above zero in at least three of the four cells.
5. The pooled gain times the 106,584 scored rows C1 corrects in v65, over 344,841, against 256.6145: at least 0.1 s.

A confirms or rejects Heathrow July for the next candidate. B only confirms F on July; January has no fresh local days,
so a pass of B leads to a January confirmation (download) before F enters any candidate.

## Result, 29 September 2026: both parts fail rule 2 only

`fresh_july.py`, `results/fresh_july.json`; all seven days (20, 24, 28, 30 July re-fetched after a network drop; the
script refused to run on the incomplete set first). Comparator bit-identical.

**A, Heathrow July:** 896 rows, RMSE 251.58 -> 226.12 s, +12,161 per row, **-5,483 without the best 100 (fails)**; halves
+6,189 / +17,796; degraded (817 rows) +11,632, well observed (79) +17,631; six of seven days positive; planning +1.50 s
(2026 mix weighted, sister-month strata where the fresh sample is thin). Fail. Heathrow July stays at the base.

**B, C1 + CatBoost blend:** 10,872 rows, +320 per row over C1, **-178 without the best 100 (fails)**; halves +357 / +275;
all four cells positive (EDDF +553, EHAM +310, LSZH +150, LEBL +102); six of seven days positive; planning +0.19 s.
Fail, as on B1 and B2. Three independent samples now agree: the blend's gain over C1 is real on average and
concentrated in about a hundred rows each time.
