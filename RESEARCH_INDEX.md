# Map of the research folders

Every folder is one question asked of the data. Code is in the `.py` files; `PREREG*.md` is the rule set written
before the run; `results/` holds the aggregate numbers. Negative results are kept, because a rejected idea is
evidence too. The first line below is each folder's own title.

| folder | what it asks | pre-reg. files | scripts | result files |
|---|---|---|---|---|
| `research_240` | Fallback lane: the neighbour-taxi anchor (v46), plus weather, runway-direction and other rejected screens | 0 | 49 | 40 |
| `research_250` | The v45 release pipeline: 91-column schema, queue and time-unit fixes, full-ensemble January/July gate | 0 | 20 | 13 |
| `research_260` | Neighbour sequence networks, the neural quarter of the ensemble from v47 | 0 | 18 | 18 |
| `research_280` | Pre-registered: copy-labelled fallback gate | 1 | 42 | 52 |
| `research_290` | Assigned-stand departure detector: frozen rules and validation plan | 2 | 16 | 14 |
| `research_300` | Geometry test: X-Plane Gateway stands against OpenStreetMap, detector and matching fixed | 4 | 20 | 63 |
| `research_310` | Label-free coverage audit of open ADS-B ground evidence for all ten airports and all scored days | 0 | 1 | 1 |
| `research_320` | Pre-registered: a second capacity step for the partial-track correction | 8 | 36 | 32 |
| `research_340_acars_formats` | Fixed-window public ACARS coverage/matching pilot. See ACARS_MATCHING_PROTOCOL.md. | 0 | 5 | 0 |
| `research_360_integration` | Integrating a stronger base under the v62 pipeline | 1 | 10 | 6 |
| `research_370_konfig` | Pre-registered: runway configuration and destination bearing as main-lane columns | 1 | 2 | 1 |
| `research_380_rome` | Pre-registered: Rome stage 2, the frozen partial-track model C1 at LIRF, unchanged | 1 | 2 | 2 |
| `research_400_cells` | Pre-registered: the frozen C1 model in three unused cells (Munich January, Madrid July, Heathrow July) | 2 | 7 | 5 |
| `research_410_surface` | Pre-registered: surface pressure, apron conflicts and relative delay at the reported off-block time | 1 | 2 | 1 |
| `research_420_fallback` | Pre-registered: count matching for Rome's copy rows in the fallback lane | 1 | 2 | 2 |
| `research_430_robust` | Pre-registered: a drift-robust partial-track correction (movement-only inputs, a second model family) | 1 | 1 | 1 |
| `research_440_fresh_july` | Pre-registered: Heathrow July and the C1 + CatBoost blend, confirmed on seven fresh July 2025 days | 2 | 5 | 4 |
| `research_450_stand_january` | Pre-registered: the frozen stand model at Zurich January and Amsterdam January, on ten unread January 2025 days | 2 | 5 | 4 |
| `research_460_fallback_family` | Pre-registered: a second model family in the fallback lane (CatBoost), blended with the shipped fallback | 1 | 2 | 2 |
| `research_470_three` | Pre-registered: Heathrow July stand rows, Amsterdam detector rows, Paris July partial rows | 1 | 1 | 1 |
| `research_480_weights` | Pre-registered: C1 at weight 1.25, and C1 averaged with the Amsterdam January add-on | 1 | 4 | 3 |
| `research_490_second` | Pre-registered: a second partial-track model on sampling-aware features, averaged with C1 | 1 | 2 | 1 |
| `research_500_midnight` | Pre-registered: hedge Rome's midnight-crossing fallback rows the model calls normal | 2 | 7 | 5 |
| `research_510_rome_classifier` | Pre-registered: a Rome-specific label-mechanism classifier for the late fallback rows | 1 | 3 | 3 |
| `research_520_rome_main` | Pre-registered: a Rome copy classifier for the main lane | 1 | 1 | 1 |
| `research_530_pushback` | Why the frozen v2 stand detector runs late away from Munich: a diagnostic on the first December 2025 block. | 0 | 1 | 1 |
| `research_540_fallback_adsb` | Callsign-free matching for rows without AOBT_3: exploratory probe on the nine December 2025 days (their labels were | 0 | 1 | 1 |
| `research_550_review` | Pre-registered: the partial-track correction C1 trained on more days | 8 | 22 | 20 |
| `research_560_holds` | Pre-registered: a ground-hold nowcast for rows without AOBT_3 (outside Rome) | 1 | 1 | 1 |
| `research_570_recorded` | Pre-registered: +24 h at Rome as "a time was recorded", learned from every late row | 1 | 1 | 1 |
| `research_580_corrector` | Pre-registered: a residual corrector on the out-of-fold base, for the rows no ADS-B layer corrects | 1 | 1 | 1 |
| `research_590_stand_new` | Pre-registered: a drift-robust stand model for Madrid and Heathrow July | 6 | 12 | 10 |
| `research_600_drift_audit` | Label-free drift audit of every deployed layer cell (v77): the v56 drift rule (partial_coverage_2026.drift) between | 0 | 1 | 1 |
| `research_610_more_data` | Pre-registered: more training data, round two, verified by rotation | 1 | 1 | 1 |
| `research_620_new_family` | Pre-registered: a second model family (ExtraTrees) blended into the base where the base is final | 1 | 1 | 1 |
| `research_630_c1_family` | Pre-registered: a second model family for C1 (CatBoost), verified by rotation | 1 | 1 | 1 |
| `research_640_tail_rule` | Pre-registered: CatBoost as a second family for C1+ and K2+, on the fresh days | 2 | 9 | 8 |
| `research_650_eham_cat` | Pre-registered: CatBoost as second family inside the Amsterdam January add-on | 1 | 3 | 3 |
| `research_660_rome_arrivals` | Pre-registered: arrival recording signals for the Rome copy classifier (v70 slice) | 1 | 1 | 1 |
| `research_670_fallback_lfpg` | Pre-registered: the v2 detector on rows without AOBT_3 at Paris (LFPG) | 1 | 1 | 1 |
| `research_680_rome_overnight` | Candidate note: Rome overnight delays, the +24 h coin flip hedged at its midpoint | 0 | 3 | 2 |
| `research_690_anatomy` | Error anatomy of the 2025 held-out predictions and base coverage of v80 (descriptive, no fit) | 0 | 3 | 0 |
| `research_700_probe` | Leaderboard experiments of 2 October: policy, scripts and ledger | 0 | 12 | 0 |
| `research_710_rome_calibration` | Pre-registered: recalibrated expected values for Rome's rows without AOBT_3 | 1 | 1 | 1 |
| `research_720_retention` | Pre-registered: gate-retention columns at the real push-back clock as main-lane columns | 1 | 2 | 2 |
| `research_730_adsb_learner` | Pre-registered: ADS-B ground tracks as per-flight features in a second-stage corrector (pilots 1 to 15b), and the 2026 rebuild | 17 | 21 | 16 |
| `research_740_ensemble_holdout` | The frozen v45 ensemble re-trained with other held-out month pairs, for the corrector tests | 0 | 1 | 0 |
| `research_800_final` | Final phase: four-month base, assembly, February/June checks (see its README) | 0 | 6 | 4 |
