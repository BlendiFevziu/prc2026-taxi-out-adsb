"""Conservative recognizer of explicit DEP clock fields, not a general decoder.

Field semantics: OAG ACARS OOOI Message Types and Examples (pages 2-3).
Recognized clocks have minute precision and NO resolved calendar date.
Unrecognized airline-specific payloads abstain; receipt time is not OUT.
"""
import re


def unique_field(text, field, pattern):
    values=re.findall(r'(?:^|[/\n\r])\s*'+field+r'\s+('+pattern+r')(?=\s|/|$)',text,re.M)
    return values[0] if values and len(set(values))==1 else None


def clock_seconds(value):
    if not value or not re.fullmatch(r'(?:[01]\d|2[0-3]):?[0-5]\d',value):return None
    digits=value.replace(':','')
    return int(digits[:2])*3600+int(digits[2:])*60


def parse_dep(text):
    """Recognize only a marked DEP with explicit flight/route/OUT fields.

    Returns structured candidate or None; no implied date, ground truth or match.
    """
    if not isinstance(text,str):return None
    text=text.upper().strip()
    if not re.search(r'(?:^|\n)\s*DEP\s*(?:\n|$)',text):return None
    # Estimated/requested times must not be promoted to actual observations.
    if re.search(r'\b(?:ESTIMATED|EXPECTED|REQUESTED|EOBT|TOBT)\b',text):return None
    flight=unique_field(text,'FI',r'[A-Z0-9]{3,10}')
    origin=unique_field(text,'DA',r'[A-Z]{4}')
    destination=unique_field(text,'DS',r'[A-Z]{4}')
    clocks={}
    for key,names in (('out',('OT','OUT')),('off',('OF','OFF'))):
        found=[]
        for name in names:
            # Capture any token so invalid or conflicting duplicates fail closed.
            vals=re.findall(r'(?:^|[/\n\r])\s*'+name+r'\s+([^/\s]+)',text,re.M)
            found.extend(vals)
        values=[clock_seconds(v) for v in found]
        if found and (None in values or len(set(values))!=1):return None
        clocks[key]=values[0] if values else None
    if not all((flight,origin,destination)) or clocks['out'] is None:return None
    return {'format':'explicit_DEP_fields','flight':flight,'origin':origin,
            'destination':destination,**clocks,'precision_s':60,'event_date':None,
            'candidate_only':True}


def parse_out_report(text):
    """Documented OUTRP/OFFRP header plus explicit clocks; never resolve its day.

    The header's day is a nominal flight-day token, not a proven event date.
    A report flight number alone is not sufficient to identify a challenge row.
    """
    if not isinstance(text,str):return None
    text=text.upper().strip()
    matches=list(re.finditer(r'(?:^|\n)\s*-?\s*\d{4}\s+(OUTRP|OFFRP)\s+'
        r'([A-Z0-9]{1,10})\s*/\s*(\d{2})\s+([A-Z]{4})/([A-Z]{4})(?=\s|$)',text))
    if len(matches)!=1:return None
    kind,flight,day,origin,destination=matches[0].groups()
    if not 1<=int(day)<=31:return None
    clocks={}
    for field in ('OUT','OFF'):
        values=re.findall(r'(?:^|[/\n\r])\s*'+field+r'\s+([^/\s]+)',text,re.M)
        times=[clock_seconds(v) for v in values]
        if values and (None in times or len(set(times))!=1):return None
        clocks[field.lower()]=times[0] if times else None
    if clocks['out'] is None or (kind=='OFFRP' and clocks['off'] is None):return None
    return {'format':kind,'report_flight':flight,'nominal_flight_day':int(day),
            'origin':origin,'destination':destination,**clocks,'precision_s':60,
            'event_date':None,'candidate_only':True}
