# Pre-registration: surface pressure, apron conflicts and relative delay at the reported off-block time

Written 29 September 2026, **before any fit with these columns**. Owner's request of the same day, after an audit found
that none of the 91 main-lane features counts aircraft on the surface, apron-level conflicts, or a flight's delay
relative to the airport's current delay level (all load features are anchored at take-off or at the same stand).
NM-free; only challenge columns. Private research: no production change, no candidate, no upload.

## Columns (`research_410_surface/features_surface.py`)

Anchored at AOBT_3, the reported off-block time of every main-lane row; a departure's own BLOCK_TIME (the unknown) is
never read, which a unit check confirms. Arrivals' in-block times are read, as `features_arrivals.py` already does.

- `gp_dep_surface`: other departures of the airport between their AOBT_3 and take-off at this flight's AOBT_3;
- `gp_arr_surface`: arrivals between landing and in-block at that moment;
- `ah_pier_moves_15`: other departures (AOBT_3) and arrivals (in-block) at stands with the same letter prefix within
  15 minutes;
- `sl_rel_delay_min`: the flight's AOBT_3 minus SCHED, minus the median of that over the airport's departures
  scheduled within 60 minutes either side.

Reference sets exclude implausible records (departures 0 to 3 h from AOBT_3 to take-off, arrivals 0 to 2 h from landing
to in-block). January 2025, label-free: built for 99.1 % of departures in 1 s; medians 9 departures and 4 arrivals on
the surface, 6 pier movements, relative delay 0 min (IQR -6 to +9).

**Prior: small.** The 28 September leads diagnostic found the main-lane tail at LIRF, LTFM and LFPG not identifiable
from comparable signals; the absolute delay version (AOBT_3 - EOBT) gave +31.7 MSE per row within a month. The anchor
AOBT_3 is itself far from the true off-block on the tail rows (median gap about 18 minutes).

## Arms, protocol, rules

As `research_390_aiu/PREREG_daily.md`: real (91 + 4), three controls (the four columns permuted jointly within airport
and calendar month), base (91); seeds 177, 277, 377; fixed 1,200 rounds; coverage precondition 95 %; **pass only if
all six rules hold** (every seed beats the mean control; both held-out months positive; positive without the best 100;
airport-day interval above zero; at least 0.3 s against the mean control; the real arm beats the base). Stage 1 pair
1_7, stage 2 pair 3_9 only if stage 1 passes. Runner: `research_410_surface/surface_screen.py`.

## Addendum A, 29 September 2026: stage 1 (pair 1_7) fails three rules

`surface_screen.py 1_7`, `results/surface_screen_1_7.json`; coverage 100 %. Seed-averaged RMSE: real 223.618, mean
control 223.778, base 223.657. Against the mean control +0.160 s (per seed +70 / +38 / +107 MSE; January +101, July +48),
against the base +0.039 s. Rules 1, 2 and 6 hold; **rule 3 fails (-72.7 without the best 100), rule 4 fails
(airport-day [-3.9, +135.8]), rule 5 fails (0.160 s below 0.3 s)**. At the hubs the columns lose (-67 per row; LIRF
-734, LTFM +46, LFPG +249); flights at least 30 minutes off the airport's current delay gain +48 against control and
lose 34 against the base. **Verdict: fail.** No stage 2; the tail at LIRF and LTFM is not reached by these columns.
