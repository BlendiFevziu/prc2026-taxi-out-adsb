# Policy: the robustness rule for tail effects (owner's decision, 1 October 2026)

From 1 October 2026 onward, pre-registered tests replace "gain above zero without the best 100 rows" by **"airport-day
block bootstrap (5,000 draws, seed 0) 95 % lower bound above zero"**. Every other rule (pooled, months, cells, seeds,
drift, planning) stays. Reason: the best-100 rule rejected effects whose gain sits in each day's largest errors (the
C1+CatBoost blend failed it on three independent blocks while its day bootstrap was [+127, +435] with no draw below
zero), and in small cells it removed up to a quarter of the rows.

The rule is **not applied retroactively**. Past near-misses are re-tested only on data not yet used for that question;
where no such data exists (Paris, Munich stand, the Amsterdam January average) they stay closed.

## Exception, 1 October 2026 (owner's decision): applied retroactively, uniformly, discounted

The owner chose to apply the bootstrap rule to past near-misses after seeing a descriptive computation for the three
that would matter (`experiments.csv`, 1 October). Applied uniformly to all three, on the data each was originally tested
on: **Paris** drift-robust C1 (702 rows, 24 airport-days, bootstrap 95 % [+3,969, +30,397]) passes; **Amsterdam January
average with C1+** (4,686 rows, 10 days, [+581, +1,924]) passes; **Munich stand** drift-robust ([-1,062, +10,647]) still
fails. Every other rule of the original registrations held for the two that pass. Because these estimates were selected
after a first look, their predictions are discounted to half the planned gain, with a worst case that allows a loss.
This is the only retroactive exception; it is recorded here, in experiments.csv and in each build report.
