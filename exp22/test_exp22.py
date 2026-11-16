"""Synthetic tests for EXPERIMENT-22 pieces (numpy/pandas parts in .venv; torch parts skip without torch)."""
import unittest

import numpy as np


class TestMetricaTruth(unittest.TestCase):
    def table(self, rows):
        import pandas as pd
        return pd.DataFrame(rows, columns=['team', 'type', 'subtype', 'period', 'start', 'end'])

    def test_rule(self):
        from exp22.metrica import truth
        periods = np.r_[np.ones(100, int)]
        ev = self.table([
            ('H', 'SET PIECE', 'KICK OFF', 1, 5, 5),
            ('H', 'PASS', '', 1, 5, 15),
            ('A', 'CHALLENGE', 'GROUND-WON', 1, 20, 20),     # duel: no change
            ('A', 'BALL LOST', 'HEAD', 1, 25, 30),           # touch without control: no change
            ('A', 'RECOVERY', '', 1, 40, 40),                # A gains control
            ('A', 'BALL OUT', '', 1, 45, 50),                # out at 50 -> D until the restart
            ('H', 'SET PIECE', 'THROW IN', 1, 60, 60),
            ('H', 'PASS', '', 1, 60, 70),
        ])
        lab = truth(ev, periods)
        self.assertTrue((lab[:5] == 'X').all())
        self.assertTrue((lab[5:40] == 'H').all())
        self.assertTrue((lab[40:50] == 'A').all())
        self.assertTrue((lab[50:60] == 'D').all())
        self.assertTrue((lab[60:71] == 'H').all())
        self.assertTrue((lab[71:] == 'X').all())                 # after the last event, no stoppage: X

    def test_fault_and_setpiece_gap(self):
        from exp22.metrica import truth
        periods = np.ones(60, int)
        ev = self.table([
            ('H', 'PASS', '', 1, 0, 8),
            ('A', 'CHALLENGE', 'TACKLE-FAULT-LOST', 1, 10, 10),
            ('H', 'FAULT RECEIVED', '', 1, 10, 10),
            ('H', 'SET PIECE', 'FREE KICK', 1, 30, 30),
            ('H', 'PASS', '', 1, 30, 40),
            ('A', 'BALL LOST', 'OFFSIDE', 1, 45, 48),
        ])
        lab = truth(ev, periods)
        self.assertTrue((lab[:10] == 'H').all())
        self.assertTrue((lab[10:30] == 'D').all())
        self.assertTrue((lab[30:48] == 'H').all())             # A's offside touch keeps H (A not in possession)
        self.assertTrue((lab[48:] == 'D').all())               # stoppage is the last event: D to period end


class TestViews(unittest.TestCase):
    def test_mirror(self):
        from exp22.views import mirror
        p = np.array([[[.2, .3], [.7, .9]]])
        t = np.array([0, 1])
        b = np.array([[.1, .4]])
        l = np.array(['H'])
        p2, t2, b2, l2 = mirror(p, t, b, l)
        np.testing.assert_allclose(p2, [[[.8, .7], [.3, .1]]])
        np.testing.assert_array_equal(t2, [1, 0])
        np.testing.assert_allclose(b2, [[.9, .6]])
        self.assertEqual(l2[0], 'A')
        self.assertEqual(l[0], 'H')                              # input untouched


try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, 'torch venv only')
class TestNet(unittest.TestCase):
    def test_blocks(self):
        from exp22.net import block_index, block_mean
        z = np.arange(24, dtype=np.float32).reshape(12, 2)
        s, e, c = block_index(0, 12)
        np.testing.assert_array_equal(s, [0, 5, 10])
        np.testing.assert_array_equal(c, [2, 7, 11])
        np.testing.assert_allclose(block_mean(z, 0, 12)[:, 0], [4, 14, 21])

    def test_short_weight(self):
        from exp22.net import short_weight
        lab = np.array(['H'] * 50 + ['A'] * 100 + ['D'] * 10 + ['H'] * 200)
        w = short_weight(lab, np.ones(len(lab), int))
        self.assertEqual(w[0], 2.)
        self.assertEqual(w[60], 1.5)
        self.assertEqual(w[155], 1.)
        self.assertEqual(w[300], 1.)

    def test_receptive_field_and_gradient(self):
        import torch
        from exp22 import net
        self.assertEqual(net.receptive_field(), 255)
        torch.manual_seed(0)
        model = net.Net(4)
        x = torch.randn(2, 4, 600)
        y = torch.randint(0, 3, (2, 600))
        batch = [x, y, torch.ones(2, 600), torch.zeros(2, 600), torch.ones(2, 600)]
        loss, _, _ = net.loss_of(model, batch)
        loss.backward()
        self.assertGreater(float(model.inp.weight.grad.abs().sum()), 0.)
        # centred receptive field: output at t depends on input at t +- 127, not at t +- 128
        model.eval()
        x = torch.zeros(1, 4, 600, requires_grad=True)
        model(x)[0][0, 0, 300].backward()
        g = x.grad[0].abs().sum(0)
        self.assertGreater(float(g[300 - 127] + g[300 + 127]), 0.)
        self.assertEqual(float(g[300 - 128] + g[300 + 128]), 0.)

    def test_predict_interp(self):
        from exp22 import net
        model = net.Net(3)
        scaler = {'med': np.zeros(3, np.float32), 'iqr': np.ones(3, np.float32), 'nan_cols': np.array([], np.int64)}
        periods = np.r_[np.ones(1003, int), 0, 2 * np.ones(997, int)]
        q, c = net.predict(model, scaler, np.random.default_rng(0).normal(size=(2001, 3)).astype(np.float32), periods)
        np.testing.assert_allclose(q.sum(1), 1, atol=1e-5)
        self.assertTrue(np.allclose(q[1003], 1 / 3) and c[1003] == 0)


if __name__ == '__main__':
    unittest.main()
