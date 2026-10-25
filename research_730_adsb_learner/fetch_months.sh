#!/bin/sh
# Fetch and extract the adsb.lol days of the given months that have no track file yet, three at a time, with the preferred-release
# tool (research_300/fetch_preferred.py: split or unsplit archives, sizes checked, raw parts deleted after extraction).
# caffeinate keeps the Mac from idle sleep while it runs (closing the lid still sleeps it; rerun this script afterwards, it skips finished days).
# Run from the repository root:  sh research_730_adsb_learner/fetch_months.sh 2025-12 2025-03 2025-05
set -e
TRACKS=../adsb-lol/tracks
LOG=.cache/research_730/fetch_months.log
DAYS=""
for month in "$@"; do
  last=$(python3 -c "import calendar,sys; y,m=map(int,sys.argv[1].split('-')); print(calendar.monthrange(y,m)[1])" "$month")
  for d in $(seq -w 1 "$last"); do
    day="$month-$d"
    if [ ! -f "$TRACKS/$day.parquet" ]; then DAYS="$DAYS $day"; rm -rf "../adsb-lol/raw/$day"; fi
  done
done
echo "$(date +%H:%M) to fetch:$DAYS" | tee -a "$LOG"
[ -n "$DAYS" ] || exit 0
nohup caffeinate -is sh -c "printf '%s\n' $DAYS | xargs -n1 -P3 python3 -u -B research_300/fetch_preferred.py" >> "$LOG" 2>&1 &
echo "started; count with: ls $TRACKS | grep -c '^2025-'"
