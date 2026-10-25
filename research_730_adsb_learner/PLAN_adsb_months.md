# Plan: which ADS-B months to fetch next, and in what order (draft of 15 October 2026)

Nothing here is started. Depends on the pilot (`pilot_screen.py`) passing its six rules; if it fails, this plan is dropped.

## What we hold (extracted tracks in `../adsb-lol/tracks/`, counted 15 October)

| 2025 | days held | days missing |
|---|---:|---:|
| January | 31 | 0 |
| February | 6 | 22 |
| March, April, May | 0 | 92 |
| June | 6 | 24 |
| July | 31 (+ extra files) | 0 |
| August to November | 0 | 122 |
| December | 9 | 22 |
| **2025 total** | **83** | **282** |

2026 (the scored months, for the final-phase analogue): January 31, February 28, June 29, July 35. Complete enough.

## Cost per day and in total (from our own fetch tool and from another team's README, not yet measured here)

About 3.3 to 3.5 GB read per day in 2 GB parts; the extracted track file is about 12 MB, and the raw parts are deleted as soon as a
day is written. 282 days is about 0.95 TB read and 3.4 GB of disk (53 GB free). Another team quotes 4.8 hours per 30-day month on one
worker and 1.6 hours on three. That would be about 47 hours for 282 days on one worker, about 16 on three. Our connection speed decides it,
so the first step measures one day before anything else.

## Order, and why

The final phase scores January, February, June and July, so the 2025 analogues of all four are the validation set.

1. **February and June 2025 (46 days).** We hold 6 days of each. This completes all four analogue months with full ADS-B and lets us
   validate the way the leading teams did (four whole months, held out in pairs).
2. **Pilot on four months** (same code as `pilot_screen.py`, folds over Jan, Feb, Jun, Jul): confirms the gain on the months that
   matter and measures the learning curve.
3. **December 2025 (22 days)**, then **March and May (62 days)**: the months adjacent to February and June. The other team's learning curve
   says neighbouring months dominate and off-season months add little.
4. **April, August to November (92 + 122 days), only if step 3 still gains** at least 0.3 s over step 2. Stop where the learning curve flattens.

## How

`../adsb-lol/fetch_days.py --dates <range> --phase fetch --workers 3` downloads with resume and a completeness marker per day;
`--phase extract --workers 4` extracts under the shared heavy-work lock and deletes the raw parts. Fetch is network-bound and can run next
to a fit; extraction is CPU-bound and must not. Releases: `planes-readsb-prod-0` as used so far (our September check found the
preferred staging release adds under 0.1 % usable aircraft). Known failure modes in the tool's own notes: stalled transfers (resumed
automatically), a truncated second part mistaken for a short day (guarded), three-part days in summer 2026.

## Gates before each step

- Step 1 starts with one day timed end to end, then a day-count check against the file list; stop if one day takes more than 30 minutes.
- After each month: row counts per airport against the neighbouring months, to catch a half-extracted day.
- Attribution (ODbL, adsb.lol contributors) stays in any release.

## State on 17 October 2026, 22:30

Fetched with `fetch_months_loop.sh` (rounds that wait for the network; `../adsb-lol/fetch_days.download` now resumes a partial file across calls and waits
60 s between retries, because the network dropped for minutes every 30 to 40 minutes and no day finished otherwise; two days in parallel). Held now: January 31,
February 28, March 31, May 31, June 30, July 31, December 30 of 31 (2025-12-31 is in neither the preferred-release list nor prod-0: HTTP 404). Missing: April,
August to November. Throughput once fixed: about 7.5 minutes per day. Next per the plan: the corrector trained on more months, the learned-stand re-count
(registered for six or more months, thresholds unchanged), and a fresh-month test of the daily reconciliation without LIRF.

## State on 19 October 2026, 20:30

Step 3 is evaluated (pilot 9, `PREREG_more_months.md`): five months beat two by 1.48 s, so step 4's gate (0.3 s) is met and April and August to November (152 days) are being fetched with
`fetch_months_loop.sh 2025-04 2025-08 2025-09 2025-10 2025-11`, about 19 hours at the measured 7.5 minutes per day. Caveat from pilot 11: in the build's form (corrector over the ensemble) the extra months only
pay off with the ensemble's own held-out predictions for them, which do not exist yet.

## State on 24 October 2026

All 2025 months held: 363 of 365 days (missing 2025-10-14, whose published archive fails to decompress on two independent downloads, and 2025-12-31, which has no release). Pilots 11 to 13 found that extra months do not help the corrector over the ensemble, so step 4's months serve as validation months, not as corrector training.
