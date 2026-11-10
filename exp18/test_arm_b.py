"""Synthetic tests for EXPERIMENT-18 arm B. No data files.

    .venv/bin/python -m unittest exp18.test_arm_b
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp18 import arm_b as ab  # noqa: E402


class Features(unittest.TestCase):
    def setUp(self):
        self.periods = np.r_[np.zeros(5, int), np.ones(200, int), 2 * np.ones(200, int)]
        n = len(self.periods)
        self.x = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 90))
        self.f = ab.features(self.x, self.periods)

    def test_shape(self):
        self.assertEqual(self.f.shape, (len(self.periods), 180))
        self.assertEqual(self.f.dtype, np.float32)

    def test_diff_nan_at_boundaries_finite_inside(self):
        for k, d in enumerate((12, 25, 50)):
            col = self.f[:, 90 + 30 * k]
            lo, hi = 5, 205
            self.assertTrue(np.isnan(col[lo:lo + d]).all() and np.isnan(col[hi - d:hi]).all())
            self.assertTrue(np.isfinite(col[lo + d:hi - d]).all())
            self.assertTrue(np.isnan(col[hi:hi + d]).all() and np.isnan(col[-d:]).all())
            self.assertTrue((col[lo + d:hi - d] == 2 * d).all())

    def test_base_columns_and_order(self):
        np.testing.assert_array_equal(self.f[:, :90], self.x)
        self.assertEqual(self.f[100, 90 + 30 + 7], 50)       # d=25, 8th diff column
        self.assertEqual(self.f[100, 90 + 60], 100)          # d=50, first diff column


class Rows(unittest.TestCase):
    def test_rule(self):
        n = 40
        labels = np.array(['H'] * n)
        labels[10] = 'X'
        periods = np.ones(n, int)
        periods[:3] = 0
        visible = np.ones(n, bool)
        visible[30] = False
        dfl = np.flatnonzero(ab.row_mask(labels, periods, visible, False))
        sc = np.flatnonzero(ab.row_mask(labels, periods, visible, True))
        self.assertEqual(list(dfl), [5, 15, 20, 25, 30, 35])
        self.assertEqual(list(sc), [5, 15, 20, 25, 35])


if __name__ == '__main__':
    unittest.main()
