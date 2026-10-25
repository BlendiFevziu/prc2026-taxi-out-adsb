#!/bin/sh
# Resume the February and June 2025 adsb.lol downloads after an interruption (sleep, no network).
# Fetches only the days that have no extracted track file yet; finished days are never touched.
# February: research_300/fetch_dates.py (split prod-0 archives). June: research_300/fetch_preferred.py, three at a time
# (June's days are single unsplit files under their preferred release). Logs go to .cache/research_730/.
# Run from the repository root:  sh research_730_adsb_learner/resume_febjun.sh
set -e
TRACKS=../adsb-lol/tracks
LOG=.cache/research_730
pkill -f "fetch_dates.py 2025-02" 2>/dev/null || true
pkill -f "fetch_preferred.py 2025-06" 2>/dev/null || true
pkill -f "xargs -n1 -P3 python3 -u -B research_300/fetch_preferred.py" 2>/dev/null || true
sleep 2
FEB=""; JUN=""
for d in $(seq -w 1 28); do [ -f "$TRACKS/2025-02-$d.parquet" ] || FEB="$FEB 2025-02-$d"; done
for d in $(seq -w 1 30); do [ -f "$TRACKS/2025-06-$d.parquet" ] || JUN="$JUN 2025-06-$d"; done
echo "missing February:$FEB"
echo "missing June:$JUN"
for d in $JUN; do rm -rf "../adsb-lol/raw/$d"; done      # half-downloaded June parts restart cleanly
if [ -n "$FEB" ]; then nohup python3 -u -B research_300/fetch_dates.py $FEB >> $LOG/fetch_feb_resume.log 2>&1 & fi
if [ -n "$JUN" ]; then nohup sh -c "printf '%s\n' $JUN | xargs -n1 -P3 python3 -u -B research_300/fetch_preferred.py" >> $LOG/fetch_jun_resume.log 2>&1 & fi
echo "started; progress: ls $TRACKS | grep -c '2025-02\|2025-06'  (58 when complete)"
