"""Synthetic alignment tests: a simulated second view with offset, subset visibility, replays, mirror."""
import unittest

import numpy as np

from v8 import align as al
from v8.interp import fill_gaps
from v8.simulate import second_view


def toy_match(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    periods = np.r_[np.ones(half, int), 0, np.full(n - half - 1, 2)]
    players = np.zeros((n, 22, 2))
    pos = rng.uniform(.1, .9, (22, 2))
    for i in range(n):
        pos = np.clip(pos + rng.normal(0, .002, (22, 2)), .02, .98)
        players[i] = pos
    ball = np.zeros((n, 2))
    b = np.array([.5, .5]); v = np.zeros(2)
    for i in range(n):
        v = .9 * v + rng.normal(0, .004, 2)
        b = np.clip(b + v, .02, .98)
        ball[i] = b
    players[half], ball[half] = np.nan, np.nan
    return players, ball, periods


class AlignTests(unittest.TestCase):
    def test_recovers_offset_mirror_and_fuses_ball(self):
        players, ball, periods = toy_match()
        bp, bb, truth = second_view(players, ball, periods)
        res = al.align(players, bp, periods, sample=50, stride=5)
        self.assertEqual(res['flip'], (True, False))
        for h in (1, 2):
            self.assertIsNotNone(res['offsets'][h])
            self.assertLessEqual(abs(res['offsets'][h] - 1234), 2)
        live = (truth >= 0) & (np.isfinite(bp).all(axis=2).sum(axis=1) >= al.MIN_PLAYERS)
        aligned = res['mapping'][live] >= 0
        self.assertGreater(aligned.mean(), .85)
        exact = res['mapping'][live][aligned] == truth[live][aligned]
        self.assertGreater(exact.mean(), .95)
        wrong_live = (res['mapping'] >= 0) & live & (res['mapping'] != truth)
        self.assertLess(wrong_live.mean(), .02)
        median, count = al.agreement(ball, bb, res)
        self.assertLess(median, 1.)
        tb = ball.copy()
        tb[::3] = np.nan  # tactical loses a third of ball frames
        fused, source = al.fuse_ball(tb, bb, res)
        recovered = (source == 1)
        self.assertGreater(recovered.sum(), .2 * np.isnan(tb[:, 0]).sum())
        err = np.linalg.norm((fused[recovered] - ball[recovered]) * al.PITCH, axis=1)
        self.assertLess(np.median(err), 1.)

    def test_chamfer_and_shots(self):
        pb = np.array([[10., 10.], [20., 20.]])
        pt = np.array([[[10., 10.], [25., 20.], [np.nan, np.nan]], [[0., 0.], [0., 0.], [0., 0.]]])
        np.testing.assert_allclose(al.chamfer(pb, pt), [2.5, 10.])
        b = np.full((100, 6, 2), np.nan)
        b[10:60] = .5
        b[80:90] = .5
        self.assertEqual(al.shots(b), [(10, 60)])

    def test_fill_gaps(self):
        periods = np.r_[np.ones(10, int), 0, np.full(9, 2)]
        ball = np.tile(np.arange(20.)[:, None], (1, 2)) / 20
        ball[[2, 3, 6, 7, 8, 9, 12]] = np.nan          # gaps: 2 (fillable), 4 (too long at K=3), 1 after the break
        filled, source = fill_gaps(ball, periods, 3)
        np.testing.assert_allclose(filled[2], [.1, .1]); np.testing.assert_allclose(filled[3], [.15, .15])
        self.assertTrue(np.isnan(filled[6:10]).all())    # gap crosses the half break: never filled
        np.testing.assert_allclose(filled[12], [.6, .6])
        self.assertEqual(source.tolist(), [0, 0, 2, 2, 0, 0, -1, -1, -1, -1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0])
        same, src0 = fill_gaps(ball, periods, 0)
        np.testing.assert_array_equal(np.isnan(same), np.isnan(ball)); self.assertEqual((src0 == 2).sum(), 0)


if __name__ == '__main__':
    unittest.main()
