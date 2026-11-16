"""Second-half pre-kickoff frames are D (standard library + numpy; no labels, no match data).

The organisers' FAQ ("One trap worth knowing"): the second half is graded from video frame 110,808 but does not kick
off until 113,742, and those 2,934 graded frames are all D. Both numbers come from the vendored organiser code: the
start of the second-half graded block from `games.half_video_blocks`, the kickoff from the organiser's
`starter/load_smart_tagging.py` (KICKOFF_VIDEO_FRAMES, "Both numbers are published, in DATA.md and FAQ.md").
games.py's `kickoff_2nd` anchor is a playback position and lands one frame earlier (113,741; the first-half anchor
lands one frame after the first graded frame), so it is a cross-check only.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter-repo'))


def window(game='BAR-ATH'):
    """(first graded frame of the second half, second-half kickoff frame), both video frames."""
    from games import GAMES, anchor_video_frame, half_video_blocks
    from starter.load_smart_tagging import GAME, KICKOFF_VIDEO_FRAMES
    assert game == GAME, (game, GAME)
    start = {half: first for half, first, _ in half_video_blocks(game)}[2]
    kickoff = KICKOFF_VIDEO_FRAMES['2nd Half']
    assert abs(anchor_video_frame(GAMES[game]['kickoff_2nd']) - kickoff) <= 1
    assert (start, kickoff, kickoff - start) == (110808, 113742, 2934), (start, kickoff)
    return start, kickoff


def kickoff_fix(labels, game='BAR-ATH'):
    """Copy of `labels` (one per video frame) with frames start..kickoff-1 set to D; every other frame unchanged."""
    from games import GAMES
    labels = np.asarray(labels)
    assert labels.shape == (GAMES[game]['n_video_frames'],), labels.shape
    start, kickoff = window(game)
    out = labels.copy()
    out[start:kickoff] = 'D'
    return out
