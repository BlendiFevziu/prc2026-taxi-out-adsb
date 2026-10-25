# Pre-registration: C1 at weight 1.25, and C1 averaged with the Amsterdam January add-on

Written 29 September 2026, **before any fit and before any row named "confirmation" below is read**. Owner's request of
the same day. NM-free. No upload without the owner.

## Test 1: C1 at weight 1.25 instead of 1.0

C1's weight was raised from 0.5 to 1.0 in v58 (+1.03 s); above 1.0 it was never tested, while the old NM add-on's own
optimum was 1.237. In v67 C1 corrects 106,584 scored rows. The stand model is left out: every stand row of the fresh
days was read in today's weight test, so no unread stand data remain.

- **Development:** the partial rows of the capacity study's B1 and B2 (`volume_capacity`, seven C1 cells; read at weight
  1.0 before, never at 1.25).
- **Confirmation (fresh):** partial rows of the ten January 2025 days downloaded today (3, 10, 11, 12, 15, 17, 22, 27,
  29, 31 January) in the four January C1 cells (EDDF, EGLL, LEBL, LSZH); only their stand rows were read today.
- Gain = squared error at weight 1.0 minus squared error at 1.25; floor 300; detector rows excluded.

Rules, all must hold: B1 pooled gain above zero; B2 pooled gain above zero; confirmation pooled gain above zero;
confirmation above zero without the 100 most favourable rows; at least three of the four confirmation cells above
zero; the confirmation gain times 106,584 over 344,841 against v67's 256.0858, at least 0.1 s.

## Test 2: Amsterdam January, the add-on averaged with C1

EHAM January partial rows (9,752 scored in 2026) carry Codex's add-on (200 x 7, 14 inputs, weight 1.0) alone. Arm:
max(base + (add-on correction + C1 correction) / 2, 300) against max(base + add-on correction, 300).

- **Development:** the eleven January days with add-on features already built (4, 14, 21, 26, 28 from Codex; 1, 2, 18;
  6, 13, 20), EHAM January partial rows, detector rows excluded. The add-on was trained on 8, 9, 24, 30 January and C1 on
  E (including 16 and 23 January); none of these days is used.
- **Confirmation (fresh):** the same rows on the ten new January days; the add-on's features are built first with
  Codex's `eham_recovery.extract` (8d98953), which asserts its own replay of category and summaries against today's
  frozen v56 extraction of those days.

Rules, all must hold: development pooled gain above zero; confirmation pooled gain above zero; confirmation above zero
without the 100 most favourable rows; at least seven of ten confirmation days above zero; the confirmation gain times
9,752 over 344,841 against 256.0858, at least 0.05 s.

Passing tests enter one candidate, built and verified like v67.

## Result, 29 September 2026: test 1 fails, test 2 passes

`weights.py`, `results/weights.json`; comparator bit-identical in every block.

**Test 1 (C1 at 1.25):** B1 -815 per row (16,029 rows), B2 -239 (8,894), fresh January -708 (12,745 rows), -1,267
without the best 100, all four cells and all ten days negative, planning -0.43 s. **Fail**: 1.0 stays.

**Test 2 (Amsterdam January, add-on averaged with C1):** development 5,098 rows +3,763 per row, +2,237 without the best
100, all eleven days positive; fresh confirmation 4,686 rows **+4,521**, +2,902 without the best 100, **all ten days
positive**; planning +0.25 s. **Pass.** The add-on's features for the fresh days were built with Codex's extractor, every
partial row replaying the frozen extraction exactly.

## Candidate v68, built and verified (29 September 2026)

`build_v68.py`, `verify_v68.py`: v67 with the add-on averaged with C1 on the 9,752 Amsterdam January rows the add-on
corrects. Every one of them reproduced its v67 value exactly from the add-on alone before the change; the verifier
recomputed both corrections from the seed models directly. 9,690 rows change, RMS 32.8 s; everything else byte-identical
to v67; NM-free. **Prediction fixed before upload: central 255.89, range 256.01 to 255.79.**
