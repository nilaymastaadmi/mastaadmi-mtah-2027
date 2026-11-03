"""Synthetic v4 tests: no external dataset, competition asset or label is read."""
import json
import unittest

import numpy as np

from v4 import develop as dv, infer
from v4.shape import team_shape


def formation(n, home_x, away_x, spread_home, spread_away, flip_second_half=False):
    rng = np.random.default_rng(0)
    half = n // 2
    periods = np.r_[np.ones(half, int), 0, np.full(n - half - 1, 2)]
    players = np.zeros((n, 4, 2))
    players[:, :2, 0] = home_x + rng.normal(0, spread_home, (n, 2))
    players[:, 2:, 0] = away_x + rng.normal(0, spread_away, (n, 2))
    players[..., 1] = .5 + rng.normal(0, .1, (n, 4))
    if flip_second_half:
        players[half + 1:, :, 0] = 1 - players[half + 1:, :, 0]
    players[half] = np.nan
    return players, np.array([0, 0, 1, 1]), periods


class V4Tests(unittest.TestCase):
    def test_shape_uses_attacking_frame_in_both_halves(self):
        players, teams, periods = formation(400, .3, .7, .05, .02, flip_second_half=True)
        x, names = team_shape(players, teams, periods)
        cx = x[:, names.index('home_cx')]
        # Home defends low x in half 1 and high x in half 2; in its own frame it sits ~0.3*105 m out.
        np.testing.assert_allclose(np.nanmean(cx[periods == 1]), .3 * 105, atol=1.)
        np.testing.assert_allclose(np.nanmean(cx[periods == 2]), .3 * 105, atol=1.)
        self.assertTrue(np.isnan(x[200]).all() or np.isnan(cx[200]))
        self.assertGreater(np.nanmean(x[:, names.index('diff_sx')]), 0)  # home more spread out

    def test_columns_per_variant(self):
        x37, shape = np.zeros((5, 37)), np.ones((5, 23))
        self.assertEqual(dv.columns(x37, shape, 'A').shape, (5, 8))
        self.assertEqual(dv.columns(x37, shape, 'B').shape, (5, 31))
        self.assertEqual(dv.columns(x37, shape, 'C').shape, (5, 60))

    def test_fold_assignment_is_by_sorted_index(self):
        self.assertEqual([dv.fold_of(['c', 'a', 'b'], m) for m in 'abc'], [0, 1, 2])

    def test_freeze_check_rejects_changes(self):
        rt = {'python': '3.13.16', 'numpy': '2.5.4', 'pandas': '3.1.0', 'sklearn': '1.9.1'}
        freeze = {'variant': 'C', 'setting': [1., .5], 'model_sha256': 'm', 'source_hashes': {'a': '1'},
                  'runtime': dict(rt, python='3.13.18')}
        sel = {'gate': {'passed': True}, 'variant': 'C', 'best': {'C': {'setting': [1., .5]}}}
        self.assertEqual(infer.check_freeze(freeze, sel, 'm', {'a': '1'}, rt), ('C', (1., .5)))
        for args in [(freeze, dict(sel, gate={'passed': False}), 'm', {'a': '1'}, rt),
                     (freeze, sel, 'x', {'a': '1'}, rt), (freeze, sel, 'm', {'a': '2'}, rt),
                     (freeze, sel, 'm', {'a': '1'}, dict(rt, sklearn='1.8')),
                     (freeze, dict(sel, best={'C': {'setting': [.75, .5]}}), 'm', {'a': '1'}, rt)]:
            with self.assertRaises(ValueError):
                infer.check_freeze(*args)

    def test_frozen_model_runs_on_synthetic_arrays(self):
        model, variant, setting = infer.load_frozen()
        players, teams, periods = formation(300, .3, .7, .05, .02)
        ball = np.random.default_rng(1).uniform(.2, .8, (300, 2))
        ball[150] = np.nan
        from v2 import methods as mt
        from v3 import features as f3
        obs = mt.observe(players, teams, ball, periods)
        labels = infer.predict(model, f3.build(players, teams, ball, periods),
                               team_shape(players, teams, periods)[0], obs['owner'], periods, variant, setting)
        self.assertTrue(set(labels[periods > 0]) <= set('HAD'))
        self.assertEqual(labels[150], 'X')


if __name__ == '__main__':
    unittest.main()
