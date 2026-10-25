# Pre-registration: count matching for Rome's copy rows in the fallback lane

Written 29 September 2026, **before any fit**. Owner's request of the same day. NM-free, challenge data only. No
production change, no candidate, no upload.

## Why

`outlier_counts.py` (same folder): the fallback lane's predicted outliers have the right size but too low a count,
and only at LIRF. 2025 (six pairs, out of sample): 627 LIRF fallback rows are late (versp = MVT - SCHED above 2 h), 396
of them are copies (|BLOCK - SCHED| <= 60 s, so the label is versp to within a minute), and the shipped model takes the
copy value on 322. In 2026 the late share at LIRF rises (42.1 to 47.3 %) while the committed share falls (51.4 to
45.3 %). A leading contestant's public tip points at the same thing ("force the outliers in the submission to look as
much like the training set outliers as possible"; his lead is mostly in rows without AOBT_3).

## Rule under test

Per validation month m of each pair, at LIRF only: k = round(r x n_late(m)), with r the share of copies among late LIRF
fallback rows in the pair's learning months; the k late rows with the highest classifier probability p get the copy
value versp; every other row keeps the shipped prediction (`prod_anchor`). Nothing is refitted for the rule itself.

The classifier probability is not saved, so each pair's production fallback (`research_240/fallback_anchor_production.py`,
production encoding, the scored recipe) is refitted once on the cached tables. **Replication gate:** the refit's
prediction must reproduce the saved `prod_anchor` (RMSE difference at most 0.01 s per pair, as in that script's own
gate); otherwise the test stops.

## Rules, pooled over the six pairs, all rows of the fallback lane

1. Pooled MSE gain against `prod_anchor` above zero.
2. Above zero after removing the **20** most favourable rows. The usual 100 would be degenerate: the rule changes only
   about a hundred rows a year (fixed now, before any fit).
3. At least four of the six pairs above zero.
4. The gain beats the mean of five random-order controls (the same k per month, rows drawn at random among the late
   rows, seeds 1 to 5), so the classifier's ranking carries it.
5. On the official scale (5,290 fallback rows of 344,841, against 258.0008) at least 0.1 s.

Reported, not judged: per pair and month, changed rows, how many new commits are copies, the planning value.

If all hold, a 2026 application follows under its own build and verification: the all-year production classifier's p on
the 383 LIRF fallback rows of 2026 (its reproduction of v63's fallback values checked first), k from the full-year 2025
rate. If any fails, the fallback lane stays as it is.

## Result, 29 September 2026: fails four of five rules

`rome_count.py`, `results/rome_count.json`. Replication exact in all six pairs (max difference 9e-12 s). Fallback RMSE
1,486.08 -> 1,523.74 s; **-113,339 MSE per fallback row, -3.37 s on the official scale**; without the best 20 rows
-202,942; pairs 2_8 +54,163, 3_9 +33,201, 4_10 +17,207, 1_7 -426,835, 5_11 -155,269, 6_12 -35,948. The five random-order
controls are worse (-249,702 to -304,239), so the ranking carries information (rule 4 holds), but committing more rows
to the copy value does not pay: 404 rows changed, 322 of them copies. A right commit gains little because the soft
prediction was already part of the way there; each of the 82 wrong commits costs an error of hours. **Verdict: fail.**
Under squared error the soft blend is the right shape; any gain in this lane has to come from a better classifier,
not from a higher count with the same one.
