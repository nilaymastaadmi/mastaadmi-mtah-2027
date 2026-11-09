# Experiment 13: broadcast-view ball fusion

Registered 4 November 2026, before any outcome. v6 stays the submission unless the gate passes and the fused pipeline is reproduced on the Mac.

## Idea

The organisers provide a second automatic tracking of the same match from the broadcast camera (164,488 frames, ball visible 50.5%, homography usable 65.8%, not time-aligned). The TV camera follows the action, so it often sees the ball when the tactical camera loses it in crowds and aerial play. Aligning the two timelines algorithmically and filling missing tactical ball frames with broadcast observations should reduce the 41% missing-ball share that drives most errors. Everything is algorithmic, using only the provided assets; no hand correction.

## Alignment (`v8/align.py`, fixed)

- Orientation: best of 4 mirror options.
- Global offset per half by Chamfer voting; shots split at cuts and capped at 30 s; per-shot local search within ±100 s; acceptance only when the mean residual ≤ 1.2 m, then per-frame residual ≤ 1.2 m.
- Fusion: a tactical frame with no ball takes the broadcast ball from the accepted frame with the lowest residual.

## Instrument checks on the competition match (label-free, Mac, `v8/broadcast_audit.py`)

All must hold or the experiment stops:

- a single offset per half is found and shots agree with it;
- ball agreement where both views see the ball: median ≤ 2 m over at least 10,000 frames;
- the aligned residual p90 ≤ 1.2 m.

Report the recovered-frame count. The diagnostic prints aggregates only.

## Evaluation on an analogue (DFL, out of fold)

Simulated second view of each DFL match from the natural data: time offset, live shots of 10–40 s with replays between (30% of time), visible players within 35 m of the ball with 10% dropout, ball visible 70% of live frames, 0.3 m noise, mirrored. Tactical input = the contested degradation; fused input = contested ball filled from the aligned simulated view.

Models: the EXPERIMENT-10 fold models retrained (same seeds, same data) so that each DFL match is scored by a model that never saw it. Compare v6 features on the contested input (reference) against the same features on the fused input, same decoder, four label conventions.

Gate: DFL contested convention-averaged mean spellF1 delta ≥ +0.02 with wins on at least 6 of 7 matches, and macroF1 delta ≥ 0.

## If the gate passes and the instrument checks pass

Freeze a fused inference entry (`v8`): load the frozen v6 model unchanged, run alignment and fusion on the competition assets, then compute features and decode. Stability checks, two identical Mac runs, and the label-free tag agreement at least v6 − 0.02. Note: the fused input has less missing ball than v6's training mixture; this is the expected direction (towards the clean regime where v6 scores higher) and is disclosed.

## Limits

- The analogue second view is simulated; real broadcast tracking has its own detector errors and homography errors (reprojection error up to 304).
- Alignment may fail on parts of the match; unaligned frames simply stay missing.
