"""Synthetic v2 tests: no external dataset, competition asset or label is read."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession import decode
from v2 import degrade as dg, experiment as ex, idsse, methods as mt


def synthetic_xml(folder):
    info = folder / 'info.xml'
    info.write_text('<?xml version="1.0"?><PutDataRequest><MatchInformation>'
                    '<Environment PitchX="100.00" PitchY="50.00"/><Teams>'
                    '<Team TeamId="G" Role="guest"/><Team TeamId="H" Role="home"/>'
                    '</Teams></MatchInformation></PutDataRequest>')
    frames = []
    for section, base in [('firstHalf', 10000), ('secondHalf', 100000)]:
        for team, person, x in [('G', 'g1', 10.), ('H', 'h1', -10.), ('REF', 'r', 0.)]:
            frames.append(f'<FrameSet GameSection="{section}" TeamId="{team}" PersonId="{person}">')
            # Home player only appears in the last two frames of each half.
            ns = range(base + 1, base + 3) if person == 'h1' else range(base, base + 3)
            frames += [f'<Frame N="{n}" X="{x}" Y="0.0"/>' for n in ns]
            frames.append('</FrameSet>')
        frames.append(f'<FrameSet GameSection="{section}" TeamId="BALL" PersonId="b">')
        for i, (poss, status) in enumerate([(1, 1), (2, 1), (2, 0)]):
            frames.append(f'<Frame N="{base + i}" X="50.0" Y="25.0" Z="0.1" '
                          f'BallPossession="{poss}" BallStatus="{status}"/>')
        frames.append('</FrameSet>')
    positions = folder / 'positions.xml'
    positions.write_text('<?xml version="1.0"?><PutDataRequest><Positions>'
                         + ''.join(frames) + '</Positions></PutDataRequest>')
    return info, positions


def toy_arrays(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    periods = np.r_[np.ones(half, int), 0, np.full(n - half - 1, 2)]
    players = rng.uniform(.2, .8, (n, 4, 2))
    ball = rng.uniform(.2, .8, (n, 2))
    players[half], ball[half] = np.nan, np.nan
    return {'players': players, 'ball': ball, 'z': rng.uniform(0, 3, n), 'periods': periods,
            'teams': np.array([0, 0, 1, 1]), 'pitch': np.array([105., 68.]),
            'labels': np.where(periods > 0, 'H', 'X')}


class V2Tests(unittest.TestCase):
    def test_viterbi_matches_frozen_v1_decoder(self):
        rng = np.random.default_rng(1)
        costs = rng.exponential(.05, (900, 3))
        periods = np.r_[np.ones(400, int), np.zeros(5, int), np.full(495, 2)]
        for penalty in [0., .05, .3, 2.]:
            np.testing.assert_array_equal(mt.viterbi(costs, periods, penalty),
                                          decode(costs, periods, penalty))

    def test_label_mapping(self):
        poss = np.array([1, 2, 1, 2, 0])
        status = np.array([1, 1, 0, 0, 1])
        self.assertEqual(''.join(idsse.labels_from_ball(poss, status)), 'HADDX')

    def test_xml_parse_assemble(self):
        with tempfile.TemporaryDirectory() as tmp:
            info, positions = synthetic_xml(Path(tmp))
            pitch, home, guest = idsse.parse_info(info)
            self.assertEqual((pitch, home, guest), ((100., 50.), 'H', 'G'))
            a = idsse.assemble(idsse.parse_positions(positions, home, guest), home, guest, pitch)
        self.assertEqual(''.join(a['labels']), 'HADXHAD')
        np.testing.assert_array_equal(a['periods'], [1, 1, 1, 0, 2, 2, 2])
        np.testing.assert_array_equal(a['teams'], [0, 1])  # home slot first, official ignored
        self.assertTrue(np.isnan(a['players'][0, 0]).all())  # home absent in first frame
        np.testing.assert_allclose(a['players'][1, 0], [.4, .5])
        np.testing.assert_allclose(a['players'][1, 1], [.6, .5])
        np.testing.assert_allclose(a['ball'][0], [1., 1.])
        self.assertTrue(np.isnan(a['ball'][3]).all())

    def test_noncontiguous_ball_frames_fail(self):
        parsed = {s: {'ball': np.array([[1, 0, 0, 0, 1, 1], [3, 0, 0, 0, 1, 1]], float),
                      'players': {}} for s in idsse.SECTIONS}
        with self.assertRaises(ValueError):
            idsse.assemble(parsed, 'H', 'G', (105., 68.))

    def test_degradation_deterministic_and_calibrated(self):
        a = toy_arrays(60000)
        p1, b1 = dg.degrade(a, 'M', 'gaps', 'eval')
        p2, b2 = dg.degrade(a, 'M', 'gaps', 'eval')
        np.testing.assert_array_equal(b1, b2)
        np.testing.assert_array_equal(p1, p2)
        _, b3 = dg.degrade(a, 'M', 'gaps', 'train')
        self.assertFalse(np.array_equal(np.isnan(b1), np.isnan(b3)))
        graded = a['periods'] > 0
        share = np.mean(np.isnan(b1[graded]).any(axis=1))
        self.assertGreaterEqual(share, .41)
        self.assertLess(share, .55)
        nat_p, nat_b = dg.degrade(a, 'M', 'natural', 'eval')
        np.testing.assert_array_equal(nat_b, a['ball'])
        self.assertTrue((a['labels'] == np.where(a['periods'] > 0, 'H', 'X')).all())

    def test_contested_mask_prefers_crowded_frames(self):
        a = toy_arrays(60000, seed=3)
        crowded = (np.arange(len(a['ball'])) % 10000) < 2000
        a['z'] = np.where(crowded, 2., 0.)
        graded = a['periods'] > 0
        _, ball = dg.degrade(a, 'M', 'contested', 'eval')
        missing = np.isnan(ball).any(axis=1)
        _, plain = dg.degrade(a, 'M', 'gaps', 'eval')
        plain_missing = np.isnan(plain).any(axis=1)
        excess = missing[crowded & graded].mean() - missing[~crowded & graded].mean()
        plain_excess = plain_missing[crowded & graded].mean() - plain_missing[~crowded & graded].mean()
        self.assertGreater(excess, .15)
        self.assertGreater(excess, plain_excess + .1)

    def test_place_runs_respects_eligibility(self):
        eligible = np.zeros(5000, bool)
        eligible[1000:4000] = True
        out = dg.place_runs(eligible, np.ones(5000), .3, np.random.default_rng(0), 2., .5, 10.)
        self.assertFalse(out[~eligible].any())
        self.assertGreaterEqual(out[eligible].mean(), .3)

    def test_hold_majority_carries_through_missing(self):
        periods = np.ones(12, int)
        obs = {'owner': np.array([0, 0, 0, -1, -1, -1, 1, 1, 1, -1, -1, -1]),
               'outside': np.zeros(12, bool)}
        self.assertEqual(''.join(mt.hold_majority(obs, periods, 0)), 'HHHHHHAAAAAA')
        obs['outside'][9:] = True
        self.assertEqual(''.join(mt.hold_majority(obs, periods, 0)), 'HHHHHHAAADDD')

    def test_misaligned_keeps_marginals_and_shifts(self):
        rng = np.random.default_rng(2)
        q = rng.dirichlet(np.ones(3), 101)
        periods = np.r_[np.ones(50, int), 0, np.full(50, 2)]
        m = mt.misaligned(q, periods)
        for half in [1, 2]:
            rows = periods == half
            np.testing.assert_allclose(np.sort(m[rows], axis=0), np.sort(q[rows], axis=0))
            self.assertFalse(np.allclose(m[rows], q[rows]))
        np.testing.assert_array_equal(m[50], q[50])

    def test_full_match_scoring_keeps_halves_apart(self):
        gt = np.array(list('HHHHAAAA') + ['X'] + list('AAAAHHHH'))
        self.assertEqual(ex.score(gt, gt)['spellF1'], 1.)
        self.assertEqual(ex.score(gt, gt)['gt_spells'], 4)
        shifted = np.roll(gt, 4)
        self.assertLess(ex.score(gt, shifted)['spellF1'], 1.)

    def test_selection_tie_rules(self):
        rows = [{'method': 'f', 'setting': s, 'condition': 'contested', 'spellF1': v}
                for s, v in [(1, .300), (2, .304), (3, .290)]]
        self.assertEqual(ex.select(rows, 'f', [1, 2, 3]), (1, .3))
        means = {'majority': .30, 'hold_majority': .305, 'hmm_hand': .2,
                 'blend_uniform': .31, 'blend_learned': .309}
        self.assertEqual(ex.choose_candidate(means), 'majority')
        means['blend_learned'] = .33
        self.assertEqual(ex.choose_candidate(means), 'blend_learned')

    def test_gates(self):
        rows = []
        for i, m in enumerate('ABCD'):
            for c in dg.CONDITIONS:
                for method, s in [('v1_frozen', .30), ('hmm_hand', .33 if i < 3 else .29),
                                  ('blend_learned', .36), ('blend_uniform', .33),
                                  ('blend_misaligned', .33)]:
                    rows.append({'match': m, 'condition': c, 'method': method,
                                 'spellF1': s, 'macroF1': .7})
        gate = ex.promotion_gate(rows, 'hmm_hand')
        self.assertTrue(gate['checks']['contested_wins_ge_3of4'])
        self.assertTrue(gate['passed'])
        self.assertTrue(ex.learned_edge_gate(rows, 'hmm_hand')['passed'])
        for r in rows:
            if r['method'] == 'blend_uniform' and r['match'] != 'A':
                r['spellF1'] = .37
        self.assertFalse(ex.learned_edge_gate(rows, 'hmm_hand')['passed'])

    def test_v2_freeze_check(self):
        from v2.verified_inference import check_v2_freeze
        rt = {'python': '3.13.16', 'numpy': '2.5.4', 'pandas': '3.1.0', 'sklearn': '1.9.1'}
        prov = {'runtime': dict(rt, python='3.13.18')}
        dev = {'candidate': 'blend_learned', 'model_sha256': 'm', 'source_hashes': {'a': '1'},
               'selected': {'blend_learned': [.75, .5]}}
        held = {'candidate': 'blend_learned', 'model_sha256': 'm', 'source_hashes': {'a': '1'},
                'promotion_gate': {'passed': True}}
        self.assertEqual(check_v2_freeze(dev, held, prov, 'm', {'a': '1'}, rt), (.75, .5))
        bad = [(dev, dict(held, promotion_gate={'passed': False}), prov, 'm', {'a': '1'}, rt),
               (dev, held, prov, 'other', {'a': '1'}, rt),
               (dev, held, prov, 'm', {'a': '2'}, rt),
               (dev, held, prov, 'm', {'a': '1'}, dict(rt, sklearn='1.8.0')),
               (dev, held, prov, 'm', {'a': '1'}, dict(rt, python='3.12.9')),
               (dict(dev, candidate='hmm_hand'), dict(held, candidate='hmm_hand'), prov, 'm',
                {'a': '1'}, rt)]
        for args in bad:
            with self.assertRaises(ValueError):
                check_v2_freeze(*args)

    def test_frozen_v2_pipeline_on_synthetic_arrays(self):
        import hashlib
        import json
        import pickle
        folder = ex.OUT
        dev = json.loads((folder / 'dev.json').read_text())
        raw = (folder / 'model.pkl').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), dev['model_sha256'])
        model = pickle.loads(raw)  # project-trained, hash-verified
        a = toy_arrays(3000, seed=9)
        obs = mt.observe(a['players'], a['teams'], a['ball'], a['periods'])
        q = mt.learned_probs(model, mt.learned_inputs(a['players'], a['teams'], a['ball'], a['periods']))
        labels = mt.blend(q, mt.prior50(obs, a['periods']), a['periods'],
                          *dev['selected']['blend_learned'])
        self.assertTrue(set(labels[a['periods'] > 0]) <= set('HAD'))
        self.assertTrue((labels[a['periods'] == 0] == 'X').all())


if __name__ == '__main__':
    unittest.main()
