"""Six fixed, small CC0 assets; aggregate schema/timing feasibility, no raw archive."""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from airframes_probe import request

ROOT=Path(__file__).resolve().parent
DATES=('2025-07-06','2026-01-06','2026-07-06')
AIRPORTS=('EDDM','EHAM','EDDF','EGLL','LEBL','LSZH','LFPG','LIRF','LEMD','LTFM')


def unpack(payload):
    with tempfile.TemporaryFile() as stream:
        stream.write(payload);stream.seek(0)
        proc=subprocess.Popen(['/opt/homebrew/bin/zstd','-d','-q','-c'],stdin=stream,
                              stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        try:
            decoded=proc.stdout.read(20_000_001)
            if len(decoded)>20_000_000:raise ValueError('Decoded asset exceeds bound')
            if proc.wait(timeout=10):raise ValueError('zstd rejected asset')
            return decoded
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            proc.stdout.close()


def dictionaries(obj,path='',depth=0):
    if depth>8:return
    if isinstance(obj,dict):
        yield path,obj
        for k,v in obj.items():yield from dictionaries(v,path+'.'+str(k),depth+1)
    elif isinstance(obj,list):
        for v in obj[:10]:yield from dictionaries(v,path+'[]',depth+1)


def summary(decoded):
    counts=collections.Counter();labels=collections.Counter();keys=collections.Counter()
    airports=collections.Counter();shapes=collections.Counter();clock_keys=collections.Counter()
    for line in decoded.splitlines():
        if not line.strip():continue
        counts['json_rows']+=1
        try:row=json.loads(line)
        except ValueError:counts['invalid_json']+=1;continue
        text_values=[]
        for path,d in dictionaries(row):
            for k,v in d.items():
                keys[path+'.'+k]+=1
                if k.lower() in ('text','msg_text') and isinstance(v,str):text_values.append(v)
                if k.lower()=='label' and isinstance(v,str):labels[v]+=1
                if re.fullmatch(r'(actual_?)?(out|off|on|in)(_?time)?',k,re.I):clock_keys[path+'.'+k]+=1
        if text_values:counts['rows_with_message_text']+=1
        text='\n'.join(text_values)
        for ap in AIRPORTS:
            if re.search(r'\b'+ap+r'\b',text):airports[ap]+=1
        for word in ('OUT','OFF','ON','IN'):
            if re.search(r'\b'+word+r'\b',text):counts[word+'_word_rows']+=1
            hits=list(re.finditer(r'\b'+word+r'\b\s*[:=/ -]?\s*(?:[01]\d|2[0-3]):?[0-5]\d(?!\d)',text,re.I))
            if hits:counts[word+'_valid_HHMM_candidate_rows']+=1
            for hit in hits[:1]:
                shape=re.sub(r'\d','#',hit.group().upper());shapes[shape]+=1
    return {'counts':dict(counts),'labels':dict(labels),'field_paths':dict(keys),
            'airport_mentions':dict(airports),'clock_field_paths':dict(clock_keys),
            'clock_token_shapes':dict(shapes)}


def main():
    inventory=json.loads((ROOT/'acars_archive_inventory.json').read_text())
    result={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'selection':
        'For each fixed date and ACARS/VDL2: first sorted zstd asset starting between 10:00 and 13:00 UTC; no substitution.',
        'raw_messages_persisted':False,'flight_matching_performed':False,'assets':[]}
    saved=ROOT/'acars_public_sample.json'
    if saved.exists():
        prior=json.loads(saved.read_text())
        assert prior['selection']==result['selection']
        result=prior
    for day in DATES:
        repo='adsblol/aircraft-data-links-'+('2026' if day=='2026-07-06' else '2025')
        release=inventory['repositories'][repo]['/releases/tags/adsblol-adl-'+day]['releases'][0]
        assert 'CC0 1.0' in release['body']
        for channel in ('acars','vdl2'):
            prefix='adsblol-adl-'+channel+'_'+day.replace('-','')+'-'
            assets=sorted((a for a in release['assets'] if a['name'].startswith(prefix)
                and a['name'].endswith('.jsonl.zst') and '100000'<=a['name'][len(prefix):len(prefix)+6]<'130000'),key=lambda a:a['name'])
            if not assets:result['assets'].append({'day':day,'channel':channel,'missing_fixed_window':True});continue
            asset=assets[0]
            previous=[x for x in result['assets'] if x.get('name')==asset['name']]
            if previous:
                assert len(previous)==1
                if previous[0].get('sha256') and asset.get('digest'):
                    assert asset['digest']=='sha256:'+previous[0]['sha256']
                print('Reusing saved aggregate',day,channel,flush=True);continue
            if asset['bytes']>2_000_000:
                result['assets'].append({'day':day,'channel':channel,'name':asset['name'],
                    'bytes':asset['bytes'],'skipped':'Fixed asset exceeds 2 MB bound; no substitution or download'})
                continue
            time.sleep(1.2)
            status,headers,payload=request(asset['url'])
            if status!=200:raise RuntimeError('Download status '+str(status))
            assert len(payload)==asset['bytes']
            digest=hashlib.sha256(payload).hexdigest()
            if asset.get('digest'):assert asset['digest']=='sha256:'+digest
            decoded=unpack(payload)
            item={'day':day,'channel':channel,'name':asset['name'],'url':asset['url'],
                  'bytes':len(payload),'decoded_bytes':len(decoded),'sha256':digest,
                  'github_digest_verified':bool(asset.get('digest')),'summary':summary(decoded)}
            result['assets'].append(item)
            (ROOT/'acars_public_sample.json').write_text(json.dumps(result,indent=2)+'\n')
            print(day,channel,item['summary']['counts'],flush=True)
    result['complete']=True
    (ROOT/'acars_public_sample.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
