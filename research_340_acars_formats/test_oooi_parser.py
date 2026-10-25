import unittest
from oooi_parser import parse_dep,parse_out_report,clock_seconds
from acars_public_sample import summary


class ParserTests(unittest.TestCase):
    def message(self,clock='OT 1015',extra=''):
        return 'DEP\nFI ZZ123/AN X-TEST/DA EDDM/DS LIRF/'+clock+extra
    def test_explicit(self):
        r=parse_dep(self.message(extra='/OF 1031'))
        self.assertEqual((r['out'],r['off'],r['origin']),(36900,37860,'EDDM'))
        self.assertIsNone(r['event_date'])
    def test_midnight_not_resolved(self):
        r=parse_dep(self.message('OUT 23:55','/OFF 0005'))
        self.assertEqual((r['out'],r['off']),(86100,300))
        self.assertIsNone(r['event_date'])
    def test_bad_clocks(self):
        for clock in ('2500','2361','12000','--','12:99'):
            self.assertIsNone(parse_dep(self.message('OT '+clock)))
    def test_conflicts(self):
        self.assertIsNone(parse_dep(self.message(extra='/OUT 1016')))
        self.assertIsNone(parse_dep(self.message(extra='/DA EGLL')))
    def test_agreeing_duplicates(self):
        self.assertIsNotNone(parse_dep(self.message(extra='/OUT 1015')))
    def test_estimates_and_free_text(self):
        for text in ('ETA\nFI ZZ123/DA EDDM/DS LIRF/OUT 1015',
                     'Please report OUT 1015',self.message(extra='/EOBT 1015')):
            self.assertIsNone(parse_dep(text))
    def test_required_identity(self):
        for part in ('FI ZZ123','/DA EDDM','/DS LIRF'):
            self.assertIsNone(parse_dep(self.message().replace(part,'')))
    def test_bounds(self):
        self.assertEqual(clock_seconds('0000'),0)
        self.assertEqual(clock_seconds('2359'),86340)
        self.assertIsNone(clock_seconds(None))
    def test_off_report(self):
        r=parse_out_report('- 1101 OFFRP 0123/31 EDDM/LIRF X-TEST\n/OUT 2355/OFF 0005/ETA')
        self.assertEqual((r['out'],r['off'],r['nominal_flight_day']),(86100,300,31))
        self.assertIsNone(r['event_date'])
    def test_out_report(self):
        r=parse_out_report('- 1001 OUTRP 0123/06 EDDM/LIRF X-TEST\n/OUT 1015')
        self.assertEqual(r['out'],36900)
        self.assertIsNone(r['off'])
    def test_padded_report_header(self):
        r=parse_out_report('1101 OFFRP  ZZ0123    /06 EDDM/LIRF X-TEST\r\n/OUT 1015/OFF 1031')
        self.assertEqual((r['report_flight'],r['out'],r['off']),('ZZ0123',36900,37860))
    def test_bad_report(self):
        for text in ('- 1101 OFFRP 0123/06 EDDM/LIRF X-TEST\n/OUT 1015',
                     '- 1001 OUTRP 0123/00 EDDM/LIRF X-TEST\n/OUT 1015',
                     '- 1001 OUTRP 0123/06 EDDM/LIRF X-TEST\n/OUT 1015/OUT 1016',
                     'OUTRP 0123/06 EDDM/LIRF X-TEST\n/OUT 1015'):
            self.assertIsNone(parse_out_report(text))
    def test_summary_nested_and_no_raw_fields(self):
        r=summary(b'{"vdl2":{"acars":{"msg_text":"OUT 1015 EDDM","label":"H1","flight":"ZZ123"}}}\n')
        self.assertEqual(r['counts']['OUT_valid_HHMM_candidate_rows'],1)
        self.assertEqual(r['airport_mentions'],{'EDDM':1})
        self.assertNotIn('ZZ123',str(r))


if __name__=='__main__':unittest.main()
