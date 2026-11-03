"""Synthetic v3 tests: no external dataset, competition asset or label is read."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from v2.methods import viterbi
from v3 import decode as d3, features as f3, skillcorner as sc


class V3Tests(unittest.TestCase):
    def test_viterbi2_matches_uniform_and_respects_asymmetry(self):
        rng = np.random.default_rng(0)
        c = rng.exponential(.05, (3000, 3))
        per = np.r_[np.ones(1500, int), 0, np.full(1499, 2)]
        for p in (0., .2, 1.):
            np.testing.assert_array_equal(d3.viterbi2(c, per, p, p), viterbi(c, per, p))
        # A brief A blip inside H is removed by a large team penalty but kept when cheap.
        costs = np.tile([0., 1., 1.], (60, 1))
        costs[20:30] = [1., 0., 1.]
        one = np.ones(60, int)
        self.assertNotIn('A', d3.viterbi2(costs / 25, one, 5., 5.))
        self.assertIn('A', d3.viterbi2(costs / 25, one, .01, 5.))

    def test_window_helpers_stay_within_half(self):
        per = np.r_[np.ones(5, int), 0, np.full(5, 2)]
        v = np.r_[np.ones(5), np.nan, np.full(5, 3.)]
        r = f3.roll_mean(v, 10, per)
        np.testing.assert_allclose(r[:5], 1.)
        np.testing.assert_allclose(r[6:], 3.)
        vals = np.array([np.nan, 2, np.nan, np.nan, 5, np.nan, np.nan, 7, np.nan, np.nan, np.nan])
        fwd = f3.carry(vals, per)
        np.testing.assert_array_equal(np.isnan(fwd[:2]), [True, False])
        self.assertEqual(fwd[3], 2.)
        self.assertTrue(np.isnan(fwd[6]))  # separator frame is not part of a half
        self.assertEqual(f3.carry(vals, per, backward=True)[2], 5.)
        vis = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], bool)
        np.testing.assert_allclose(f3.time_from_visible(vis, per)[:5], [0, .04, .08, 0, .04])
        np.testing.assert_allclose(f3.time_from_visible(vis, per, backward=True)[6:9], [.08, .04, 0])

    def test_feature_shape_and_missing_handling(self):
        rng = np.random.default_rng(1)
        n = 600
        per = np.r_[np.ones(300, int), 0, np.full(299, 2)]
        players = rng.uniform(.2, .8, (n, 4, 2))
        ball = rng.uniform(.2, .8, (n, 2))
        ball[100:200] = np.nan
        players[300], ball[300] = np.nan, np.nan
        x = f3.build(players, np.array([0, 0, 1, 1]), ball, per)
        self.assertEqual(x.shape, (n, 37))
        self.assertTrue(np.isnan(x[150, 8]))       # distance difference unknown while missing
        self.assertFalse(np.isnan(x[150, 18]))     # carried last-visible difference
        self.assertFalse(np.isnan(x[150, 19]))     # carried next-visible difference
        self.assertAlmostEqual(x[150, 16], (150 - 99) / 25)   # time since last visible
        self.assertAlmostEqual(x[150, 17], (200 - 150) / 25)  # time until next visible

    def test_skillcorner_upsample_and_labels(self):
        v = np.array([[0.], [10.], [np.nan], [30.]])
        up = sc.upsample(v, 8)  # positions 0, .4, .8, 1.2, 1.6, 2.0, 2.4, 2.8
        np.testing.assert_allclose(up[:3, 0], [0, 4, 8])
        self.assertTrue(np.isnan(up[3, 0]) and np.isnan(up[5, 0]))
        self.assertEqual(''.join(sc.nearest(np.array(list('HADH')), 8)), 'HHAADDDH')

    def test_skillcorner_parse_synthetic_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta, track = Path(tmp) / 'meta', Path(tmp) / 'track'
            (meta / '7').mkdir(parents=True)
            track.mkdir()
            info = {'home_team': {'id': 1}, 'pitch_length': 100, 'pitch_width': 50,
                    'players': [{'id': 11, 'team_id': 1}, {'id': 22, 'team_id': 2}],
                    'match_periods': [{'period': 1, 'start_frame': 0, 'end_frame': 4},
                                      {'period': 2, 'start_frame': 5, 'end_frame': 9}]}
            (meta / '7' / '7_match.json').write_text(json.dumps(info))
            rows = []
            for f in range(10):
                rows.append({'frame': f, 'period': 1 if f < 5 else 2,
                             'ball_data': {'x': 0., 'y': 0., 'is_detected': f % 2 == 0},
                             'possession': {'group': 'home team' if f < 3 else None},
                             'player_data': [{'player_id': 22, 'x': 10., 'y': 0.},
                                             {'player_id': 11, 'x': -10., 'y': 0.}]})
            (track / '7_tracking_extrapolated.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            a = sc.load(7, meta, track)
        n1 = int(np.floor(4 * 25 / 10)) + 1
        self.assertEqual(len(a['periods']), 2 * n1 + 1)
        np.testing.assert_array_equal(a['teams'], [0, 1])  # home slot first
        np.testing.assert_allclose(a['players'][0, 0], [.4, .5])
        np.testing.assert_allclose(a['ball'][0], [.5, .5])
        self.assertTrue(np.isnan(a['ball'][1]).all())   # odd source frame undetected
        self.assertEqual(a['labels'][0], 'H')
        self.assertEqual(a['labels'][n1 - 1], 'D')
        self.assertEqual(a['labels'][n1], 'X')


if __name__ == '__main__':
    unittest.main()
