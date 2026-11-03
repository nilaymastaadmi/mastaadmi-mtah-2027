import unittest
import xml.etree.ElementTree as ET

import numpy as np

from first_assessment import epts_layout, stress
from external_validation import measure


class AssessmentTests(unittest.TestCase):
    def test_substitution_layout_uses_references_not_metadata_order(self):
        xml = '''<main><Score idLocalTeam="H" idVisitingTeam="A"/>
        <Player id="a" teamId="A"/><Player id="h" teamId="H"/><Player id="h2" teamId="H"/>
        <PlayerChannel id="ax" playerId="a" channelId="x"/><PlayerChannel id="ay" playerId="a" channelId="y"/>
        <PlayerChannel id="hx" playerId="h" channelId="x"/><PlayerChannel id="hy" playerId="h" channelId="y"/>
        <PlayerChannel id="h2x" playerId="h2" channelId="x"/><PlayerChannel id="h2y" playerId="h2" channelId="y"/>
        <DataFormatSpecification startFrame="1" endFrame="10"><SplitRegister>
        <SplitRegister><PlayerChannelRef playerChannelId="hx"/><PlayerChannelRef playerChannelId="hy"/></SplitRegister>
        <SplitRegister><PlayerChannelRef playerChannelId="ax"/><PlayerChannelRef playerChannelId="ay"/></SplitRegister>
        </SplitRegister></DataFormatSpecification>
        <DataFormatSpecification startFrame="11" endFrame="20"><SplitRegister>
        <SplitRegister><PlayerChannelRef playerChannelId="ax"/><PlayerChannelRef playerChannelId="ay"/></SplitRegister>
        <SplitRegister><PlayerChannelRef playerChannelId="h2x"/><PlayerChannelRef playerChannelId="h2y"/></SplitRegister>
        </SplitRegister></DataFormatSpecification></main>'''
        _, specs, teams = epts_layout(ET.fromstring(xml))
        self.assertEqual(specs, [(1, 10, [1, 0]), (11, 20, [0, 2])])
        self.assertEqual(teams.tolist(), [1, 0, 0])
        with self.assertRaises(ValueError):
            epts_layout(ET.fromstring(xml.replace('startFrame="11"', 'startFrame="12"')))
        with self.assertRaises(ValueError):
            epts_layout(ET.fromstring(xml.replace('playerChannelId="hy"', 'playerChannelId="ay"')))

    def test_stress_block_length_and_shared_seed(self):
        ball = np.zeros((10000, 2))
        for block in [10, 50, 250]:
            masked = stress(ball, block)
            self.assertEqual(np.isnan(masked[:, 0]).sum(), 4100 if block != 250 else 4000)
            self.assertTrue(np.allclose(masked, stress(ball, block), equal_nan=True))
            self.assertTrue(np.all(np.isnan(masked[:, 0]).reshape(-1, block).std(axis=1) == 0))

    def test_shifted_metric_control_is_worse_than_perfect(self):
        gt = np.array(list('H' * 50 + 'D' * 50 + 'A' * 50 + 'D' * 50 + 'H' * 50))
        periods = np.ones(len(gt), int)
        perfect = measure(gt, gt, periods)
        shifted = measure(gt, np.roll(gt, 25), periods)
        self.assertEqual(perfect['spellF1'], 1.)
        self.assertLess(shifted['spellF1'], perfect['spellF1'])


if __name__ == '__main__':
    unittest.main()
