# Pre-registration: a smaller corrector (pilot 15b)

Written 24 October 2026, after pilot 15 (every larger corrector was worse on January/July) and **before any fit for this test**. Changes go into dated addenda.

## 1. Grid and selection

Rounds {300, 600} x leaves {15, 31, 63}; everything else as pilot 15. Selection on January/July exactly as pilot 15 (pilot 6's day folds, mean of seeds 177 and 277, lowest RMSE). **Ties within 0.05 s of
the best go to the default 600 x 63**, then to fewer rounds, then fewer leaves.

## 2. Judgement and meaning

Exactly pilot 15's sections 2 and 3: the selected configuration against the default on February/June and November/December (643,659 rows, six rules, each of the four months positive); pass builds
`final_rebuild_2026_tuned.parquet` with it, fail keeps the current file. If the default is selected, nothing is judged.

## Addendum A, 24 October 2026: the default is selected again; nothing judged

`pilot15b_screen.py`, `results/pilot15b_screen.json`, log `.cache/research_730/pilot15b.log`. January/July selection RMSE: **600x63 200.992**, 300x63 201.332, 600x31 201.392, 300x31 202.262, 600x15 202.280,
300x15 203.861. With pilot 15, the default is the best of eleven configurations on both sides; the corrector's capacity is settled.
