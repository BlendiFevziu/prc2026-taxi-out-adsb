# Pre-registration: the Amsterdam and Munich detectors on rows without AOBT_3

Written 30 September 2026, **before any fallback row is scored**. Part of the owner's review. Reopens the 21 September
closure ("ADS-B off-block: the economics"), which rested on the old detector (389 s raw) and one day; the guarded v2
detector and a callsign-free match now exist, and no detector has ever been applied to fallback rows. NM-free, no fit.
No upload without the owner.

## Arm, fixed now

Rows: fallback departures (no AOBT_3) at EHAM and EDDM. Matching, in order: (1) the detector's flight_mvt callsign
matcher; (2) otherwise the stand-parked free match (unique visit ending within 300 s of MVT with a guarded onset at the
assigned stand), **skipping every visit claimed by any departure's callsign match at that airport and day
(plus_callsign_flt over all departures)**, the lesson of `PREREG_eham_free_match.md`, fixed here before scoring.
Estimation: the deployed configurations (X-Plane-first stands, arm_test.guarded; the tow guard at EHAM only, as in
v55). Value: MVT minus the onset midpoint, floor 300. Reference: the shipped fallback's six-pair out-of-sample value
(`prod_anchor`).

## Test

Every local 2025 day with tracks: all 31 January days, all 31 July days, nine December days (71). Their fallback labels
were read in the six-pair fallback work, never against a detector. Rules, all must hold: pooled gain above zero; above
zero without the best 20 rows; January and July each above zero; at least four of the six airport-month blocks
(EHAM, EDDM x January, July, December) with rows above zero; planning: per airport, gain per covered row times its 2025
coverage share times its 2026 fallback rows (EHAM 1,206, EDDM 238), over 344,841, against v73's central 252.76, at
least 0.1 s.

A pass leads to v74 = v73 plus the detector value on the 2026 fallback rows it covers, built and verified like the
others.

## Result, 30 September 2026: pass

`fallback_detector.py`, `results/fallback_detector.json`. 71 days, 250 covered fallback rows (137 callsign, 113 free):
**+107,091 per covered row, +10,879 without the best 20, January +396,284, July +30,855, 4 of 6 blocks, planning
+0.24 s.** EHAM 199 of 661 rows covered, detector RMSE 172 s against the shipped 358 s; EDDM 51 of 251, 82 s against
383 s. December is negative at both airports (EHAM -45,629, EDDM -40,004 per row), reported as a warning.

## Addendum A, 30 September 2026: support guard for the 2026 application, decided without any 2026 label

The first build (`build_v74_fallback.py`) changed 398 rows, 45 of them by a disagreement the 2025 test never covered:
in 2025, 235 of 250 disagreements (detector minus shipped) lay within +/-600 s, 14 in +600 to +3,600 s (detector closer
on 13), one at -600 to -3,600 s and none beyond an hour; in 2026, 33 lie in -3,600 to -600 s, 4 below -3,600 s and 8
above +3,600 s. The detector is applied only where the disagreement lies in (-600 s, +3,600 s], the range the test
supports; the other rows keep v73. This restricts the application, it does not change the tested rule, and it was fixed
after the test on the 2025 support table alone, disclosed here.

## Candidate v74, built and verified (30 September 2026)

`build_v74_fallback.py` (with addendum A), `verify_v74.py`: 398 covered 2026 fallback rows (EHAM 309, EDDM 89), 45
outside the supported range kept at v73, **353 change** (RMS 1,120 s, max 3,555 s); everything else identical to v73;
the verifier re-derived all 353 values from the tracks in a fresh process. SHA 9cb301d6... Prediction fixed in the
build: central 252.58, best 252.50, worst 252.69 (from v73's central).
