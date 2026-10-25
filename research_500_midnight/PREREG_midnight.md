# Pre-registration: Rome's long-delay fallback rows without a local-midnight crossing are copies

Written 30 September 2026, **before the rule is applied to any 2026 row and before any build**. Owner's request of 29
September ("let's look at where the others have the upper hand"). NM-free, challenge data only, no fit. No upload
without the owner.

## Why

At our level the ranking is decided by a handful of fallback rows. The gap to tenth place (254.74) is 2.4e8 squared
seconds over the whole set, one row missed by 4.3 h; the gap to the leader (238.82) is 2.95e9, one row missed by 15 h.
In 2025, 15 fallback rows carry half of all fallback squared error, and the leader says his lead is mostly in the rows
without AOBT_3.

Rome writes three kinds of block time in the fallback lane (2025, all twelve months, labels already read in earlier
work; this rule was found by looking at them, which is disclosed here):

- **copy**: BLOCK = SCHED, so the label is the whole delay MVT - SCHED;
- **+24 h**: BLOCK carries the scheduled local date with the actual local clock time, so a pushback after local
  midnight makes the label the real taxi plus 24 h;
- **normal**: the real taxi.

The +24 h stamp needs a local-midnight crossing: without one it occurs **0 times** (727 non-copy fallback rows and
126,087 main-lane rows without a crossing as the rule defines it; 731 and 126,134 with the crossing taken from the
labelled true off-block). Without a crossing and with a delay above 8 h, every Rome fallback row was a copy (**12 of
12**; above 6 h 20 of 21). The shipped model has no crossing feature and hedges toward +24 h where it is impossible
(2026: 16.29 h predicted on a 14.87 h delay).

## Rule, fixed now

Rows: 2026 departures at LIRF without AOBT_3 (the fallback lane), delay = MVT - SCHED above 28,800 s, and no local
crossing: the Europe/Rome calendar date of MVT minus 15 minutes equals the Europe/Rome calendar date of SCHED.
New value: the delay, exactly. Every other row keeps v68's value. The 8 h threshold is the lowest with no counterexample
in 2025; it is fixed here and not tuned.

## Evidence and its limit

No unread labelled Rome row remains, so there is no confirmation block; the case rests on the mechanism and on 12 of 12.
`evidence_2025.py` recomputes the 2025 tables above. The realised outcome turns on about ten rows: all copies would give
roughly +0.78 s from v68; one normal row among them could cost up to about 8.3e8 squared seconds. The build fixes the
central, best and worst values before any upload.

## Candidate v69, built and verified (30 September 2026)

`build_v69.py`, `verify_v69.py` (the rule recomputed row by row with zoneinfo): v68 plus the rule on 10 Rome fallback
rows, 9 change (one already sat at its delay); everything else byte-identical to v68; NM-free; SHA 6f125836...
**Prediction fixed before upload: central 255.32 (+0.57 s on v68's 255.89), best 255.11 (all ten copies), worst 260.31**
(the one row with the largest risk, 39,181 -> 48,841 s, is a real taxi). That row carries two thirds of the gain and
nearly all of the risk; 2025 shows 12 of 12 copies in this cell, and v68 already predicts 81 % of the delay there.
