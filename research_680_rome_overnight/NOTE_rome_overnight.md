# Candidate note: Rome overnight delays, the +24 h coin flip hedged at its midpoint

Written 1 October 2026, before any 2026 value is computed. **Post-hoc, disclosed:** the rule below was found by reading
every 2025 Rome fallback row that crosses local midnight (`crossing` of `research_500_midnight/rule.py`), labels
included. It is therefore not a tested result, and its prediction is wide. NM-free, no fit. No upload without the owner.

## What the 2025 rows show

Every one of the 12 "+24 h" rows (y = 87,168 to 88,392 s) crosses local midnight, and its BLOCK carries the actual
off-block clock time on the **scheduled** local date. Among recorded crossing rows the date looks like the one nearest
the scheduled time: the 24 crossing rows with a correct date all had delays of 1.2 to 5.0 h (off-block just after
midnight, the next day is nearer), the 11 +24 h rows with a delay of 14.1 to 26.0 h (same day nearer). One +24 h row has
a 3.3 h delay and a 23:59 schedule (placeholder-like). By delay, crossing rows, kinds in 2025:

| delay | copy | normal | +24 h |
|---|---|---|---|
| 8 to 12 h | 11 | 0 | 0 |
| 12 to 14 h | 8 | 0 | 0 |
| 14 h and more | 8 | 0 | 11 |

The shipped value for these rows is about the delay (2025 median offset +3.0 k s on copies, +1.9 k on +24 h rows; v81
in 2026 the same). Above 14 h that commits a coin flip to one side; a +24 h row then misses by 25,000 to 40,000 s.

## Rule, fixed now

Rows: Rome (LIRF) departures without AOBT_3, crossing exactly 1.

- (a) delay at least 14 h: value (delay + Y24) / 2, with Y24 the median label of the twelve 2025 +24 h rows
  (87,420.5 s). This is the expected value at p = 0.5, shrunk from 2025's 11 of 19 (0.58) toward even odds.
- (b) delay above 8 h and below 14 h: value = delay (the v69 rule extended to crossing rows; 19 of 19 copies in 2025).

Every other row keeps its v81 value.

## Prediction, fixed in the build before any upload

For (a) the change is a bet on a handful of rows, so the build reports the exact distribution over the 2^n outcomes
(each row copy or +24 h) and fixes: central at p = 0.41 (all 27 rows above 12 h in 2025), best at p = 0.58, and the
worst case "every (a) row a copy". (b) is valued at its rows being copies.

## Candidate v82, built and verified (1 October 2026)

`build_v82.py`, `verify_v82.py` (crossing recomputed row by row with zoneinfo, Y24 from the raw 2025 files): v81 plus
12 changed rows, 5 in (a) and 7 in (b); everything else identical to v81; NM-free. SHA 7cd3de07... Y24 = 87,420.5 s.
Prediction fixed in the build: **central 243.997 (p = 0.41), best 242.154 (p = 0.58), worst 248.388 (all five (a) rows
copies)**, 237.54 if all five are +24 h. Over the 32 equally likely outcomes at p = 0.5: 10th / 50th / 90th percentile
238.84 / 243.02 / 247.14. Two of the five (a) rows carry most of the stake (Jan 2 delay 15.7 h, Jul 16 delay 16.8 h).
Not uploaded. Upload only v82 or v81, once: uploading both would read these rows' labels off the board.
