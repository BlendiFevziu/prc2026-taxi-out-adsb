# Pre-registration: the v2 detector on rows without AOBT_3 at Barcelona, Zurich and Frankfurt

Written 30 September 2026, **before any row of these airports is scored**. Extends `PREREG_fallback_detector.md` (EHAM and
EDDM, passed). NM-free, no fit. No upload without the owner.

## Arm, fixed now

Rows: fallback departures (no AOBT_3) at LEBL, LSZH and EDDF. Matching exactly as in `fallback_detector.py` (the
flight_mvt callsign matcher, else the stand-parked free match skipping every visit any callsign match claims).
Estimation: arm_test.guarded (v2 plus the arrival-stay guard) with OpenStreetMap stands (the v56 stand maps at these
airports), no tow guard (none is validated here). Value max(MVT - onset, 300). Reference: `prod_anchor`.
Application to 2026, fixed now: the same support guard as addendum A of the EHAM/EDDM test, disagreement with the
current value in (-600 s, +3,600 s].

## Test

The same 71 local 2025 days. Rules per airport (each airport decided on its own), all must hold: pooled gain above
zero; above zero without the best 10 rows; January and July each above zero; at least 20 covered rows. Planning over
the passing airports: gain per covered row times 2025 coverage share times 2026 fallback rows (LEBL 442, LSZH 659,
EDDF 421), over 344,841, against v74's central, at least 0.05 s.

A passing airport enters v75 = v74 plus the detector on its covered 2026 fallback rows within the support range.

## Result, 30 September 2026: LEBL and LSZH pass, EDDF fails

`fallback_detector_more.py`, `results/fallback_detector_more.json`. LEBL: +71,147 per covered row, +13,057 without the
best 10, January +66,551, July +75,993 (December +41,930), shipped RMSE 409 s. LSZH: 58 of 303 rows covered, detector
165 s against 298 s, +61,726 per row, +17,675 without the best 10, January and July positive (December -5,969). EDDF:
67 covered, detector 327 s against 262 s, -38,553: fail. Planning over LEBL and LSZH +0.08 s (computed against v74's
first central 252.56, read before the support guard moved it to 252.58; the bar of 0.05 s is met either way).

## Candidate v75, built and verified (30 September 2026)

`build_v75.py`, `verify_v75.py`: 2026 fallback rows covered LEBL 67 and LSZH 80, applied within the support range 66
and 73; 139 change (RMS 468 s, max 2,232 s); everything else identical to v74; values re-derived in a fresh process.
SHA 60627876... Prediction fixed in the build: central 252.54, best 252.52, worst 252.56 (from v74's central).
