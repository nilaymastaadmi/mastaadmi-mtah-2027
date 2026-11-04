"""Synthetic v6 tests: no external dataset, competition asset or label is read."""
import unittest

import numpy as np

from v6 import infer
from v6.control import control_features


def runner(n=200):
    per = np.r_[np.ones(n // 2, int), 0, np.full(n - n // 2 - 1, 2)]
    players = np.full((n, 2, 2), .5)
    players[:, 1] = [.9, .9]
    players[:, 0, 0] = .3 + np.arange(n) / 25. * 3 / 105
    ball = players[:, 0].copy()
    ball[n // 2], players[n // 2] = np.nan, np.nan
    return players, np.array([0, 1]), ball, per


class V6Tests(unittest.TestCase):
    def test_control_detects_dribble_and_rejects_distant_ball(self):
        players, teams, ball, per = runner()
        x = control_features(players, teams, ball, per)
        self.assertEqual(x.shape, (200, 12))
        self.assertGreater(x[50, 2], .9)
        self.assertEqual(x[50, 3], 0.)
        self.assertEqual(x[50, 10], 1.)
        far = ball.copy()
        far[:, 1] += 5 / 68
        self.assertEqual(control_features(players, teams, far, per)[50, 2], 0.)

    def test_control_stays_within_half(self):
        players, teams, ball, per = runner()
        ball[101:] = np.nan  # no control in half 2
        x = control_features(players, teams, ball, per)
        self.assertEqual(x[150, 10], 0.)  # last-control team does not carry across halves

    def test_freeze_check_rejects_changes(self):
        rt = {'python': '3.13.16', 'numpy': '2.5.4', 'pandas': '3.1.0', 'sklearn': '1.9.1'}
        fr = {'model_sha256': 'm', 'source_hashes': {'a': '1'}, 'runtime': dict(rt, python='3.13.18')}
        infer.check_freeze(fr, {'passed': True}, 'm', {'a': '1'}, rt)
        for args in [(fr, {'passed': False}, 'm', {'a': '1'}, rt), (fr, {'passed': True}, 'x', {'a': '1'}, rt),
                     (fr, {'passed': True}, 'm', {'a': '2'}, rt), (fr, {'passed': True}, 'm', {'a': '1'}, dict(rt, numpy='1'))]:
            with self.assertRaises(ValueError):
                infer.check_freeze(*args)

    def test_frozen_model_runs_on_synthetic_arrays(self):
        model = infer.load_frozen()
        players, teams, ball, per = runner()
        from v2 import methods as mt
        from v3 import features as f3
        from v4.shape import team_shape
        obs = mt.observe(players, teams, ball, per)
        labels = infer.predict(model, f3.build(players, teams, ball, per), team_shape(players, teams, per)[0],
                               control_features(players, teams, ball, per), obs['owner'], per)
        self.assertTrue(set(labels[per > 0]) <= set('HAD'))
        self.assertEqual(labels[100], 'X')


if __name__ == '__main__':
    unittest.main()
