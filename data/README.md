# Competition data (not included)

This repository contains no competition data. The PRC Data Challenge 2026 files are issued by
the organisers (EUROCONTROL Performance Review Commission, with the OpenSky Network) to
registered teams through the challenge's MinIO buckets, and they are not redistributed here.

Place the sixteen files in this folder with their original names:

    training_2025-01-01_2025-02-01.parquet ... training_2025-12-01_2026-01-01.parquet   twelve months
    ranking.parquet            the movements to be scored in the first phase, January and July 2026
    submitting.parquet         the submission template of the first phase
    final_ranking.parquet      the movements of the final phase: January, February, June, July 2026
    final_submitting.parquet   the submission template of the final phase (670,790 departures)

With your own team access configured in the MinIO client (the alias is called `prc26` here):

    mc cp --recursive prc26/prc-2026-datasets/ data/

Then `python3 check_dataset.py` reports row counts, fill rates and month coverage.

Note on the final files: January and July appear in both ranking files with the same `MVT_ID_mvt`.
In `final_ranking.parquet` `MVT_TIME_UTC_mvt` of those rows (and, for arrivals, `BLOCK_TIME_UTC_mvt`) differs by up
to 12 seconds from `ranking.parquet` on about 92 % of the rows; all `_flt` columns and
`SCHED_TIME_UTC_mvt` are identical. `research_800_final/assemble_final.py` accounts for this.
