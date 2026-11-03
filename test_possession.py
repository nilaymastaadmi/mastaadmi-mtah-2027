import itertools
import unittest
import copy

import numpy as np
import pandas as pd

from possession import decode, features
from external_validation import proxy_labels, mask_ball, measure
from verified_inference import check_freeze


class PossessionTests(unittest.TestCase):
    def test_freeze_rejects_changed_model_settings_or_sources(self):
        dev = {'selected': [.25, .5], 'model_sha256': 'original', 'source_hashes': {'code': 'v1'}}
        held = {**copy.deepcopy(dev), 'gate': {'provisional_gate_passed': True}}
        upstream = {'model_sha256': 'original'}
        check_freeze(dev, held, upstream, 'original', {'code': 'v1'})
        with self.assertRaises(ValueError):
            check_freeze(dev, held, {'model_sha256': 'new'}, 'new', {'code': 'v1'})
        altered = copy.deepcopy(held); altered['selected'] = [.5, .5]
        with self.assertRaises(ValueError):
            check_freeze(dev, altered, upstream, 'original', {'code': 'v1'})
        with self.assertRaises(ValueError):
            check_freeze(dev, held, upstream, 'original', {'code': 'v2'})
        altered = copy.deepcopy(held); altered['gate']['provisional_gate_passed'] = False
        with self.assertRaises(ValueError):
            check_freeze(dev, altered, upstream, 'original', {'code': 'v1'})

    def test_decoder_matches_brute_force(self):
        costs = np.array([[0., 3., 4.], [2., 0., 4.], [4., 0., 3.], [4., 3., 0.]])
        penalty = .7
        sequences = list(itertools.product(range(3), repeat=len(costs)))
        def energy(seq):
            return sum(costs[i, state] for i, state in enumerate(seq)) + penalty * sum(
                seq[i] != seq[i - 1] for i in range(1, len(seq)))
        expected = min(sequences, key=energy)
        self.assertEqual(''.join(decode(costs, np.ones(4), penalty)),
                         ''.join('HAD'[s] for s in expected))

    def test_period_resets_and_ungraded(self):
        costs = np.array([[0., 1., 2.], [3., 0., 2.], [0., 0., 0.]])
        self.assertEqual(''.join(decode(costs, [1, 2, 0], 1000)), 'HAX')

    def test_missing_ball_is_neutral_between_teams(self):
        players = np.tile([[[.2, .5], [.8, .5]]], (50, 1, 1))
        ball = np.tile([.2, .5], (50, 1)); ball[20:30] = np.nan
        costs, owner = features(players, [0, 1], ball, np.ones(50))
        self.assertTrue(np.all(costs[20:30, 0] == costs[20:30, 1]))
        self.assertTrue(np.all(owner[20:30] == -1))
        self.assertTrue(np.all(decode(costs, np.ones(50), 5) == 'H'))

    def test_metres_change_nearest_player(self):
        players = np.array([[[.52, .5], [.5, .525]]])
        _, owner = features(players, [0, 1], [[.5, .5]], [1])
        self.assertEqual(owner[0], 1)  # 2.1 m vs 1.7 m

    def test_outside_pitch_supports_dead_ball(self):
        players = np.tile([[[.2, .5], [.8, .5]]], (50, 1, 1))
        costs, _ = features(players, [0, 1], np.tile([1.2, .5], (50, 1)), np.ones(50))
        self.assertTrue(np.all(decode(costs, np.ones(50), 5) == 'D'))

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            decode(np.array([[np.nan, 0, 1]]), [1])
        with self.assertRaises(ValueError):
            features(np.zeros((2, 1, 2)), [0], np.zeros((2, 2)), [1, 1])
        for penalty in [np.nan, np.inf, -1.]:
            with self.assertRaises(ValueError):
                decode(np.array([[4., 0., 3.]]), [1], penalty)
        with self.assertRaises(ValueError):
            features(np.zeros((2, 2, 2)), [0, 1], np.zeros((2, 2)), [1, 1], fps=np.nan)

    def test_invalid_event_frame_fails(self):
        events = pd.DataFrame([{'Team': 'Home', 'Type': 'BALL OUT', 'Subtype': '',
                                'Period': 1, 'Start Frame': 3, 'End Frame': 999}])
        with self.assertRaises(ValueError):
            proxy_labels(events, np.ones(8))

    def test_event_anchor_frames_and_conflicts(self):
        events = pd.DataFrame([
            {'Team': 'Home', 'Type': 'PASS', 'Subtype': '', 'Period': 1, 'Start Frame': 1, 'End Frame': 2},
            {'Team': 'Home', 'Type': 'BALL OUT', 'Subtype': '', 'Period': 1, 'Start Frame': 3, 'End Frame': 4},
            {'Team': 'Away', 'Type': 'SET PIECE', 'Subtype': '', 'Period': 1, 'Start Frame': 6, 'End Frame': 6},
        ])
        self.assertEqual(''.join(proxy_labels(events, np.ones(8))), 'HHHDDAAA')

    def test_proxy_does_not_carry_across_halves(self):
        events = pd.DataFrame([{'Team': 'Home', 'Type': 'PASS', 'Subtype': '',
                                'Period': 1, 'Start Frame': 1, 'End Frame': 2}])
        self.assertEqual(''.join(proxy_labels(events, [1, 1, 1, 2])), 'HHHX')

    def test_stress_is_deterministic_and_contiguous(self):
        ball = np.zeros((10000, 2))
        a, b = mask_ball(ball), mask_ball(ball)
        self.assertTrue(np.allclose(a, b, equal_nan=True))
        self.assertEqual(np.isnan(a[:, 0]).sum(), 4100)
        self.assertTrue(np.all(np.isnan(a[:, 0]).reshape(-1, 50).std(axis=1) == 0))

    def test_metric_positive_and_negative_controls(self):
        gt = np.array(list('HHHDDDAAAHHHDDDAAA'))
        periods = np.ones(len(gt), int)
        perfect = measure(gt, gt.copy(), periods)
        constant = measure(gt, np.full(len(gt), 'H'), periods)
        self.assertEqual(perfect['spellF1'], 1.)
        self.assertEqual(perfect['macroF1'], 1.)
        self.assertLess(constant['spellF1'], perfect['spellF1'])
        self.assertLess(constant['macroF1'], perfect['macroF1'])


if __name__ == '__main__':
    unittest.main()
