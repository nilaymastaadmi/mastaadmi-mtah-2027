"""Synthetic tests for EXPERIMENT-18 shared code. No external or competition data.

    .venv/bin/python -m unittest exp18.test_exp18
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2 import methods as mt  # noqa: E402
from exp18 import decode as de, geometry as ge, views as vw  # noqa: E402


def toy_q(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.dirichlet([1, 1, 1], size=n)
    return q


class Decoder(unittest.TestCase):
    def test_v8_setting_equals_v2_viterbi(self):
        for seed in range(5):
            n = 3000
            q = toy_q(n, seed)
            periods = np.r_[np.zeros(10, int), np.ones(1500, int), 2 * np.ones(n - 1510, int)]
            ref = mt.blend(q, np.full((n, 3), 1 / 3), periods, 1., .5)
            c = np.random.default_rng(seed + 9).random(n)
            got = de.decode(q, c, periods, .5, 0.)
            self.assertTrue(np.array_equal(ref, got))

    def test_change_evidence_lowers_switch_cost(self):
        n = 400
        q = np.tile([.6, .4, 1e-3], (n, 1))
        q[200:230] = [.4, .6, 1e-3]          # weak, short A evidence inside a long H
        periods = np.ones(n, int)
        plain = de.decode(q, None, periods, .5, 0.)
        c = np.zeros(n)
        c[198:202] = 1.
        c[228:232] = 1.
        aware = de.decode(q, c, periods, .5, 4.)
        self.assertFalse((plain == 'A').any())
        self.assertTrue((aware[200:230] == 'A').all())

    def test_d_penalty_unchanged_by_beta(self):
        n = 300
        q = np.tile([.7, .29, .01], (n, 1))
        q[100:110] = [.3, .2, .5]
        periods = np.ones(n, int)
        c = np.ones(n)
        a = de.decode(q, c, periods, .5, 4.)
        b = de.decode(q, None, periods, .5, 0.)
        self.assertEqual(int((a == 'D').sum()), int((b == 'D').sum()))


class Targets(unittest.TestCase):
    def test_change_events_skip_restarts(self):
        lab = np.array(list('HHHAAADDDHHHAA'))
        ev = vw.change_events(lab)
        self.assertEqual(list(np.flatnonzero(ev)), [3, 12])

    def test_near_stays_in_half(self):
        ev = np.zeros(10, bool)
        ev[4] = True
        periods = np.r_[np.ones(5, int), 2 * np.ones(5, int)]
        self.assertEqual(list(np.flatnonzero(vw.near(ev, 2, periods))), [2, 3, 4])


class Geometry(unittest.TestCase):
    def test_shapes_and_missing_ball(self):
        n, p = 60, 4
        rng = np.random.default_rng(1)
        players = rng.random((n, p, 2))
        teams = np.array([0, 0, 1, 1])
        ball = players[:, 0, :] + .001
        ball[20:30] = np.nan
        periods = np.r_[np.ones(30, int), 2 * np.ones(30, int)]
        x = ge.geometry(players, teams, ball, periods)
        self.assertEqual(x.shape, (n, len(ge.NAMES)))
        self.assertTrue(np.isnan(x[20:30, 1:]).all())
        self.assertTrue((x[20:30, 0] == 0).all())
        k = ge.NAMES.index('nearest_team')
        self.assertTrue((x[:20, k] == 1).all())       # home player 0 is always nearest
        self.assertTrue((x[:20, ge.NAMES.index('home_d1')] < .2).all())


if __name__ == '__main__':
    unittest.main()
