# Pre-registration: the v2 detector on rows without AOBT_3 at Heathrow and Madrid

Written 30 September 2026, **before any row of these airports is scored**. Identical design, rules and support guard as
`PREREG_fallback_detector_more.md` (LEBL and LSZH passed, EDDF failed), for EGLL (248 fallback rows in 2026) and LEMD
(238). Known beforehand and disclosed: at both airports the recorded block time precedes visible movement by minutes
(`research_530_pushback`), so the detector runs late; whether it still beats the fallback model is the question.
Planning over the passing airports against v75's central, at least 0.05 s. A passing airport enters v76 = v75 plus the
detector on its covered 2026 fallback rows within the support range.

## Result, 30 September 2026: both fail

`fallback_detector_egll_lemd.py`, `results/fallback_detector_egll_lemd.json`. EGLL: 32 of 485 rows covered, detector
348 s against the shipped 853 s, +606,951 per row pooled but **-108,297 without the best 10** (a few extreme rows carry
it). LEMD: 35 of 331 covered, 149 s against 250 s, +40,176 per row, -4,107 without the best 10, and **no covered January
row**. Neither enters; v75 stays the latest candidate.
