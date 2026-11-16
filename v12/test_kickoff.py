"""Synthetic checks for v12/kickoff.py (no match data).

    .venv/bin/python -m unittest v12.test_kickoff
"""
import unittest

import numpy as np

from v12.kickoff import kickoff_fix, window

N = 199319


class KickoffFix(unittest.TestCase):
    def test_window_from_organiser_code(self):
        self.assertEqual(window(), (110808, 113742))

    def test_sets_only_the_pre_kickoff_span_to_d(self):
        rng = np.random.default_rng(0)
        labels = rng.choice(np.array(['H', 'A', 'D', 'X']), N)
        labels[[110807, 110808, 113741, 113742]] = ['A', 'H', 'A', 'H']
        before = labels.copy()
        out = kickoff_fix(labels)
        self.assertTrue((labels == before).all(), 'input was modified')
        self.assertEqual(out.dtype, labels.dtype)
        self.assertTrue((out[110808:113742] == 'D').all())
        changed = np.flatnonzero(out != labels)
        self.assertEqual(changed.min(), 110808)
        self.assertEqual(changed.max(), 113741)
        self.assertEqual(len(changed), int((labels[110808:113742] != 'D').sum()))
        self.assertTrue((out[:110808] == labels[:110808]).all())
        self.assertTrue((out[113742:] == labels[113742:]).all())

    def test_edges_and_idempotence(self):
        labels = np.full(N, 'H')
        out = kickoff_fix(labels)
        self.assertEqual(int((out == 'D').sum()), 2934)
        self.assertEqual((out[110807], out[110808], out[113741], out[113742]), ('H', 'D', 'D', 'H'))
        self.assertTrue((kickoff_fix(out) == out).all())

    def test_refuses_wrong_length(self):
        with self.assertRaises(AssertionError):
            kickoff_fix(np.full(N - 1, 'H'))


if __name__ == '__main__':
    unittest.main()
