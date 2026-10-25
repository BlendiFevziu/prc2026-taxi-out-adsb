# Pre-registration: the v2 detector on rows without AOBT_3 at Paris (LFPG)

Written 1 October 2026, **before any LFPG fallback row is scored against a detector**. Owner: "lets go all out on the
extreme fallback rows". Same design as `research_550_review/PREREG_fallback_detector_more.md` (LEBL and LSZH passed,
EDDF failed), with the robustness rule of `research_640_tail_rule/POLICY_tail_rule.md`. NM-free, no fit. No upload
without the owner.

Why Paris: 734 of 2026's 5,290 fallback rows; in the 2025 six-pair predictions its fallback RMSE is 489 s once its five
largest rows are set aside, the highest of the airports with ADS-B ground coverage. Istanbul (721 rows) and Rome (383)
have almost no ADS-B ground points (43 and 495 per day in the cached tracks), so no detector is possible there.

## Arm, fixed now

Rows: LFPG fallback departures in the 71 cached 2025 days (`fallback_detector.DAYS`). Matching exactly as in
`fallback_detector.day_rows`: the flight_mvt callsign matcher, else the stand-parked free match skipping every visit any
`plus_callsign_flt` match claims. Estimation `arm_test.guarded` with OpenStreetMap stands (`stand_table("LFPG")`, empty
X-Plane table), no tow guard. Value max(MVT - onset, 300). Reference: `prod_anchor`.

## Rules, all must hold

Pooled gain above zero; airport-day bootstrap (5,000 draws, seed 0) 95 % lower bound above zero; January and July each
above zero; at least 20 covered rows; planning: gain per covered row times the 2025 coverage share times 734, over
344,841, against v81's central 245.020, at least 0.05 s. Reported: December, the match route, and the share of covered
rows inside the support range of addendum A (disagreement detector minus shipped in (-600 s, +3,600 s]).

A pass leads to v82 = v81 plus the detector on LFPG's covered 2026 fallback rows within that support range, built and
verified like v75.

## Result, 1 October 2026: fail, no coverage

`fallback_lfpg.py`, `results/fallback_lfpg.json`. 996 LFPG fallback rows over the 71 days, **0 covered**: no row had a
guarded onset at its stand. Consistent with the 2026 coverage audit, where only 27 LFPG main rows were ever seen at
their assigned stand (28,457 have no points before take-off): adsb.lol sees Paris departures near the runway, not at the
stands. Nothing built.
