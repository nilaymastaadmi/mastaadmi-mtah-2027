# Experiment 17, stage 2: match audit (6 November 2026). Verdict: STOP, v8 stays

Registration: `work/EXPERIMENT-17.md`. Screen: `outputs/experiment-17-screen.md`. Audit output:
`work/v9/audit_run1.json`. Label-free, aggregates only, run once on the Mac; nothing was changed in response.

Detector run 1 (`v9.video_detect`, MPS, half precision): 199,319 frames decoded, 93,222 with a blob;
detections sha256 173b0dc7….

| check (registered) | line | result | pass |
|---|---|---:|:---:|
| I1 median distance to ATD ball, frames where both exist | ≤ 1.0 m over ≥ 10,000 frames | 2.24 m over 46,739 | no |
| I2 share of those more than 3 m apart | ≤ 0.10 | 0.465 | no |
| I3 graded frames still missing the ball after the fill | ≤ 0.276 | 0.359 (from 0.376) | no |

Other aggregates: p90 distance 77.3 m; detections passing tau on 36.2% of graded frames, 39.2% of them more
than 2 m off the pitch; 2,688 graded frames filled by video (1.7%).

## What this says

- Almost half the detections that pass tau are not the tracked ball: p90 77 m and 39% off the pitch point to
  spare balls, crowd, adverts or other false blobs. The ISSIA screen warned of this (precision 0.30 when balls
  look 1.5x larger than the training scale).
- Even a clean detector would add little here: the fill reached only 1.7% of graded frames, against the 10
  points I3 asked for. The frames the tactical tracking misses are the ones the video misses too (occlusion,
  0/30 occluded on ISSIA), matching the broadcast-fusion result (3%).
- The ball-gap lever via a frame-level detector is closed. Ball gaps may still be attacked by inference from
  players (no video), which the bench will price.

v8 remains the submission (`Mastaadmi.csv`, bb0ba6f5…). No freeze file, second run or adoption steps were run.
