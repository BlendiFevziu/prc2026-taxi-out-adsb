"""Fixed-window public ACARS coverage/matching pilot. See ACARS_MATCHING_PROTOCOL.md."""
import collections
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.request

from acars_public_sample import AIRPORTS, dictionaries
from oooi_parser import parse_dep, parse_out_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'skenderbeu-nm-v54/data'
DAYS = ('2025-07-06', '2026-01-06', '2026-07-06')
UTC = dt.timezone.utc


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    return re.sub(r'\s+', '', value.upper()) if isinstance(value, str) else ''


def window(day):
    start = dt.datetime.fromisoformat(day).replace(tzinfo=UTC, hour=10).timestamp()
    return start, start + 10800


def select_assets(inventory):
    selected = []
    for day in DAYS:
        year = '2026' if day == '2026-07-06' else '2025'
        release = inventory['repositories']['adsblol/aircraft-data-links-' + year][
            '/releases/tags/adsblol-adl-' + day]['releases'][0]
        assert 'CC0 1.0' in release['body']
        start, end = window(day)
        for asset in sorted(release['assets'], key=lambda a: a['name']):
            m = re.fullmatch(r'adsblol-adl-(acars|vdl2)_(\d{8}-\d{6})_(\d{8}-\d{6})\.jsonl\.zst', asset['name'])
            if not m:
                continue
            lo, hi = [dt.datetime.strptime(x, '%Y%m%d-%H%M%S').replace(tzinfo=UTC).timestamp()
                      for x in (m[2], m[3])]
            if lo < end and hi >= start:
                assert 0 < asset['bytes'] <= 5_000_000 and asset['digest'].startswith('sha256:')
                selected.append(dict(asset, day=day, channel=m[1]))
    assert len(selected) == 58
    assert sum(a['bytes'] for a in selected) == 37_767_711
    assert sum(a['bytes'] for a in selected) <= 60_000_000
    return selected


def receipt_epoch(row):
    value = row.get('timestamp', row.get('vdl2', {}).get('t', {}).get('sec'))
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return float(value)
    return None


def resolve_clocks(event, takeoff, receipt):
    if event['off'] is None:
        return None
    midnight = int(takeoff // 86400) * 86400
    candidates = [midnight + offset + event['off'] for offset in (-86400, 0, 86400)]
    possible = [off for off in candidates if abs(off - takeoff) <= 120
                and -120 <= receipt - off <= 21600]
    if len(possible) != 1:
        return None
    off = possible[0]
    out = (off // 86400) * 86400 + event['out']
    if out > off:
        out -= 86400
    if not 0 <= off - out <= 21600:
        return None
    return out, off


def match_event(event, receipt, index):
    identity = normalize(event.get('flight') or event.get('report_flight'))
    if not re.fullmatch(r'(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{3,10}', identity):
        return 'incomplete_identity', None
    if event['off'] is None:
        return 'missing_OFF', None
    key = (event['origin'], event['destination'], identity)
    rows = index.get(key, [])
    candidates = {}
    for row in rows:
        clocks = resolve_clocks(event, row['takeoff'], receipt)
        if clocks is not None:
            candidates[row['id']] = dict(row, out=clocks[0], off=clocks[1])
    if not candidates:
        return ('clock_disagreement' if rows else 'no_route_identity_match'), None
    if len(candidates) != 1:
        return 'ambiguous_match', None
    return 'matched_report', next(iter(candidates.values()))


def consolidate(matches):
    grouped = collections.defaultdict(list)
    for row in matches:
        grouped[row['id']].append(row)
    accepted, conflicts, duplicates = [], 0, 0
    for rows in grouped.values():
        clocks = {(r['out'], r['off']) for r in rows}
        if len(clocks) > 1:
            conflicts += 1
        else:
            accepted.append(rows[0])
            duplicates += len(rows) - 1
    return accepted, conflicts, duplicates


def load_index():
    # Small column reads only, no target columns and no training process.
    with open('/private/tmp/skenderbeu-heavy-work.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        import pandas as pd
        columns = ['MVT_ID_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'ADEP_mvt',
                   'ADES_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt']
        sources = {
            '2025-07-06': DATA / 'training_2025-07-01_2025-08-01.parquet',
            '2026-01-06': DATA / 'ranking.parquet',
            '2026-07-06': DATA / 'ranking.parquet',
        }
        ids = set(pd.read_parquet(DATA / 'submitting.parquet', columns=['MVT_ID_mvt'])['MVT_ID_mvt'])
        indices, denominators = {}, {}
        for day in DAYS:
            start, end = window(day)
            lo = dt.datetime.fromtimestamp(start - 21600, UTC)
            hi = dt.datetime.fromtimestamp(end + 120, UTC)
            frame = pd.read_parquet(sources[day], columns=columns,
                filters=[('MVT_TIME_UTC_mvt', '>=', lo), ('MVT_TIME_UTC_mvt', '<', hi)])
            frame = frame[frame.PHASE_mvt.eq('DEP') & frame.ADEP_mvt.isin(AIRPORTS)]
            if day.startswith('2026'):
                frame = frame[frame.MVT_ID_mvt.isin(ids)]
            assert frame.MVT_ID_mvt.is_unique
            index = collections.defaultdict(list)
            counts, window_counts = collections.Counter(), collections.Counter()
            for r in frame.to_dict('records'):
                stamp = r['MVT_TIME_UTC_mvt'].timestamp()
                row = {'id': int(r['MVT_ID_mvt']), 'takeoff': stamp,
                       'origin': r['ADEP_mvt'], 'day': day}
                for identity in {normalize(r['FLIGHT_mvt']), normalize(r['CALLSIGN_flt'])} - {''}:
                    index[(r['ADEP_mvt'], r['ADES_mvt'], identity)].append(row)
                counts[r['ADEP_mvt']] += 1
                if start <= stamp < end:
                    window_counts[r['ADEP_mvt']] += 1
            indices[day] = index
            denominators[day] = {'eligible_departures': {ap: counts[ap] for ap in AIRPORTS},
                                 'departures_10_to_13': {ap: window_counts[ap] for ap in AIRPORTS}}
        inputs = {str(p): sha(p) for p in set(sources.values()) | {DATA / 'submitting.parquet'}}
        return indices, denominators, inputs


def downloaded_rows(asset):
    req = urllib.request.Request(asset['url'], headers={'User-Agent': 'PRC-bounded-research/1.0'})
    with urllib.request.urlopen(req, timeout=40) as response:
        payload = response.read(5_000_001)
    assert len(payload) == asset['bytes']
    assert 'sha256:' + hashlib.sha256(payload).hexdigest() == asset['digest']
    with tempfile.TemporaryFile() as stream:
        stream.write(payload)
        stream.seek(0)
        proc = subprocess.Popen(['/opt/homebrew/bin/zstd', '-d', '-q', '-c'],
                                stdin=stream, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            size = 0
            while True:
                line = proc.stdout.readline(2_000_001)
                if not line:
                    break
                size += len(line)
                assert len(line) <= 2_000_000 and size <= 160_000_000
                if line.strip():
                    yield json.loads(line)
            assert proc.wait(timeout=10) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()


def write_result(result):
    (ROOT / 'acars_matching_result.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    assets = select_assets(json.loads((ROOT / 'acars_archive_inventory.json').read_text()))
    manifest = {'assets': assets, 'hashes': {name: sha(ROOT / name) for name in (
        'ACARS_MATCHING_PROTOCOL.md', 'acars_matching.py', 'oooi_parser.py', 'acars_archive_inventory.json')}}
    (ROOT / 'acars_matching_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    indices, denominators, inputs = load_index()
    result = {'started_utc': dt.datetime.now(UTC).isoformat(), 'complete': False,
              'manifest_sha256': sha(ROOT / 'acars_matching_manifest.json'),
              'target_columns_read': False, 'models_fit': 0, 'raw_messages_retained': False,
              'input_hashes': inputs, 'denominators': denominators, 'assets': []}
    matches = []
    write_result(result)
    try:
        for number, asset in enumerate(assets, 1):
            counts, origins, formats, mentions = [collections.Counter() for _ in range(4)]
            start, end = window(asset['day'])
            for row in downloaded_rows(asset):
                counts['decoded_rows'] += 1
                receipt = receipt_epoch(row)
                if receipt is None:
                    counts['missing_numeric_receipt'] += 1
                    continue
                if not start <= receipt < end:
                    counts['outside_receipt_window'] += 1
                    continue
                counts['window_rows'] += 1
                for _, fields in dictionaries(row):
                    for key, value in fields.items():
                        if key.lower() not in ('text', 'msg_text') or not isinstance(value, str):
                            continue
                        counts['texts'] += 1
                        for ap in AIRPORTS:
                            if re.search(r'\b' + ap + r'\b', value):
                                mentions[ap] += 1
                        event = parse_dep(value) or parse_out_report(value)
                        if event is None:
                            if re.search(r'\b(?:OUT|OFF)\b\s*[:=/ -]?\s*(?:[01]\d|2[0-3]):?[0-5]\d(?!\d)', value, re.I):
                                counts['unsupported_OUT_OFF_clock_texts'] += 1
                            continue
                        counts['recognized_reports'] += 1
                        origins[event['origin']] += 1
                        formats[event['format']] += 1
                        if event['origin'] not in AIRPORTS:
                            counts['non_challenge_origin'] += 1
                            continue
                        outcome, match = match_event(event, receipt, indices[asset['day']])
                        counts[outcome] += 1
                        if match:
                            matches.append(match)
            result['assets'].append({'day': asset['day'], 'name': asset['name'],
                'digest_verified': True, 'bytes': asset['bytes'], 'counts': dict(counts),
                'report_origins': dict(origins), 'formats': dict(formats),
                'airport_mentions_not_matches': dict(mentions)})
            write_result(result)
            print(f"{number}/{len(assets)} {asset['day']} {asset['channel']}: "
                  f"{counts['window_rows']} rows, {counts['recognized_reports']} reports, "
                  f"{counts['matched_report']} matches", flush=True)
        accepted, conflicts, duplicates = consolidate(matches)
        result['summary'] = {'matched_reports': len(matches), 'conflicting_movements': conflicts,
            'duplicate_reports': duplicates, 'unique_accepted_movements': len(accepted),
            'accepted_by_day_airport': {day: {ap: sum(r['day'] == day and r['origin'] == ap
                for r in accepted) for ap in AIRPORTS} for day in DAYS}}
        # These IDs are local follow-up inputs only; no external identity query.
        # Usually empty; keeping only accepted matches permits label inspection after freezing.
        (ROOT / 'acars_matching_accepted.json').write_text(json.dumps(accepted, indent=2) + '\n')
        result['accepted_sha256'] = sha(ROOT / 'acars_matching_accepted.json')
        result['complete'] = True
        result['finished_utc'] = dt.datetime.now(UTC).isoformat()
        write_result(result)
        print(json.dumps(result['summary']), flush=True)
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        write_result(result)
        raise


if __name__ == '__main__':
    main()
