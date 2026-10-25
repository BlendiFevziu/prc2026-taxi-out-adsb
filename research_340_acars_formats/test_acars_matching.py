import datetime as dt
import unittest
from acars_matching import resolve_clocks, match_event, consolidate, receipt_epoch


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.t = dt.datetime(2025, 7, 6, 0, 5, tzinfo=dt.timezone.utc).timestamp()
        self.event = dict(flight='ZZ123', origin='EDDM', destination='LIRF', out=23*3600+55*60, off=300)
        self.row = dict(id=1, takeoff=self.t, origin='EDDM', day='2025-07-06')
        self.index = {('EDDM', 'LIRF', 'ZZ123'): [self.row]}

    def test_midnight_rollover(self):
        out, off = resolve_clocks(self.event, self.t, self.t + 60)
        self.assertEqual((off-out, off), (600, self.t))

    def test_receipt_cannot_substitute_for_event(self):
        self.assertIsNone(resolve_clocks(self.event, self.t + 600, self.t + 600))
        self.assertIsNone(resolve_clocks(self.event, self.t, self.t + 21601))
        self.assertIsNone(resolve_clocks(self.event, self.t, self.t - 121))

    def test_missing_off_and_excessive_taxi(self):
        self.assertIsNone(resolve_clocks(dict(self.event, off=None), self.t, self.t))
        self.assertIsNone(resolve_clocks(dict(self.event, out=12*3600), self.t, self.t))

    def test_exact_identity_route_and_unique_flight(self):
        self.assertEqual(match_event(self.event, self.t, self.index)[0], 'matched_report')
        self.assertEqual(match_event(dict(self.event, flight='YY123'), self.t, self.index)[0], 'no_route_identity_match')
        self.assertEqual(match_event(dict(self.event, destination='EGLL'), self.t, self.index)[0], 'no_route_identity_match')
        self.index[('EDDM', 'LIRF', 'ZZ123')].append(dict(self.row, id=2))
        self.assertEqual(match_event(self.event, self.t, self.index)[0], 'ambiguous_match')

    def test_numeric_flight_abstains(self):
        self.assertEqual(match_event(dict(self.event, flight='123'), self.t, self.index)[0], 'incomplete_identity')

    def test_duplicates_and_conflicts(self):
        row = dict(self.row, out=self.t-600, off=self.t)
        self.assertEqual(consolidate([row, row]), ([row], 0, 1))
        self.assertEqual(consolidate([row, dict(row, out=self.t-660)]), ([], 1, 0))

    def test_epoch_units_and_nested_receipt(self):
        self.assertEqual(receipt_epoch({'vdl2': {'t': {'sec': self.t}}}), self.t)
        self.assertIsNone(receipt_epoch({'timestamp': '2025-07-06'}))


if __name__ == '__main__':
    unittest.main()
