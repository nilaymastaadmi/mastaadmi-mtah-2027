# Experiment 17, stage 1: ISSIA held-out screen (6 November 2026)

Registration: `work/EXPERIMENT-17.md`. Script: `v9/issia_screen.py`; summary `work/v9-screen/summary.json`.
WASB soccer weights (d0369572…) on ISSIA-CNR cameras 5 and 6, which the weights never saw; 599 labelled
in-game ball frames (569 visible, 30 occluded), 5,399 frames with no ball in view. Half precision on the
Mac GPU; a repeat of 300 frames was identical.

## Threshold selection (registered rule)

Native scale (k = 1.0), cameras pooled:

| peak threshold | recall | precision | detections |
|---|---:|---:|---:|
| 0.5 | 0.825 | 0.946 | 522 |
| **0.6** | **0.811** | **0.980** | 496 |
| 0.7 | 0.786 | 0.983 | 479 |
| 0.8 | 0.743 | 0.985 | 452 |
| 0.9 | 0.633 | 0.990 | 383 |

tau = 0.6, the smallest threshold with precision ≥ 0.95.

## Scale robustness at tau = 0.6

| k (ball size vs native) | recall | precision | camera 5 recall / precision | camera 6 recall / precision |
|---|---:|---:|---:|---:|
| 0.75 | 0.411 | 0.957 | 0.085 / 0.744 | 0.838 / 0.995 |
| 1.0 | 0.811 | 0.980 | 0.718 / 0.964 | 0.934 / 0.996 |
| 1.5 | 0.654 | 0.296 | 0.788 / 0.515 | 0.479 / 0.154 |

Stop rules: recall ≥ 0.60 at k = 1.0 (0.811) and ≥ 0.40 at k = 0.75 (0.411). **Screen passed**, narrowly at
k = 0.75.

## What this means for the match

- Occluded balls are essentially never found: 0 of 30 at k = 1.0. If the tactical tracking lost the ball
  mostly because it was occluded, recovery on the match will be small. I3 measures this.
- The detector is sensitive to scale. The tactical view spans about 22 to 44 px/m (k ≈ 0.75 to 1.5 around
  the 29 px/m centre the tiling was set for): far-side balls will be found less often, and near-side tiles
  risk false detections (precision 0.30 at k = 1.5 on ISSIA, mostly on frames with no ball in view, which
  cannot happen on the panning tactical camera, where the ball is nearly always in frame). I1 and I2 measure
  this on the match without labels.
- Median position error of correct detections: 2.6 px at k = 1.0, about 0.05 m on ISSIA.

Nothing was changed in response to these numbers. Next: the match run, once the tactical video is on the Mac.
