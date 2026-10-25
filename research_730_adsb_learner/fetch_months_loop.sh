#!/bin/sh
# Robust version of fetch_months.sh: works in rounds. Each round waits until github.com answers, then fetches the still-missing days of the
# given months two at a time, so one day downloads while the other extracts (17 October: with three parallel downloads and no resume no day finished between the network drops every 30-40 min; downloads now resume) (research_300/fetch_preferred.py). A network outage therefore makes the next round wait instead of
# failing the whole queue in seconds (as on 17 October 2026, 150 "Could not resolve host" failures). Stops when nothing is missing, or after
# 12 rounds in a row that add no day.
# Run from the repository root:  nohup caffeinate -is sh research_730_adsb_learner/fetch_months_loop.sh 2025-12 2025-03 2025-05 &
TRACKS=../adsb-lol/tracks
LOG=.cache/research_730/fetch_months.log
missing() {
  for month in "$@"; do
    last=$(python3 -c "import calendar,sys; y,m=map(int,sys.argv[1].split('-')); print(calendar.monthrange(y,m)[1])" "$month")
    for d in $(seq -w 1 "$last"); do [ -f "$TRACKS/$month-$d.parquet" ] || printf ' %s' "$month-$d"; done
  done
}
stale=0
while true; do
  DAYS=$(missing "$@")
  [ -n "$DAYS" ] || { echo "$(date +%H:%M) all days present" >> "$LOG"; break; }
  before=$(ls "$TRACKS" | grep -c '^2025-')
  until curl -s -o /dev/null --connect-timeout 10 https://github.com; do echo "$(date +%H:%M) network down, waiting" >> "$LOG"; sleep 60; done
  # raw parts are kept between rounds: fetch_days.download resumes a partial file (17 October 2026)
  echo "$(date +%H:%M) round: missing$DAYS" >> "$LOG"
  printf '%s\n' $DAYS | xargs -n1 -P2 python3 -u -B research_300/fetch_preferred.py >> "$LOG" 2>&1
  after=$(ls "$TRACKS" | grep -c '^2025-')
  if [ "$after" -gt "$before" ]; then stale=0; else stale=$((stale + 1)); fi
  [ "$stale" -lt 12 ] || { echo "$(date +%H:%M) stopping: 12 rounds without progress" >> "$LOG"; break; }
done
