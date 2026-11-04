# Experiment 10: explicit ball-control features

Registered 3 November 2026, before any EXPERIMENT-10 outcome. v4 stays the submission unless this gate passes and the new pipeline is reproduced on the Mac.

## Motivation

- The competition defines possession by **control**: a player keeps the ball for a few seconds, and touches or deflections do not count.
- v4's features describe proximity and team shape but never measure control.
- Live-play team assignment is the dominant loss (oracle analysis), and decoding and tag fusion did not help (EXPERIMENTS 08 and 09).

## Features (`v6/control.py`, thresholds fixed now, not tuned)

| Quantity | Definition |
|---|---|
| Velocities | Players and ball, from 5-frame centred differences in metres per second; NaN across half boundaries or where positions are missing |
| Controlled frame | The ball is visible, the nearest player is within 1.5 m, and the ball moves within 2.5 m/s of that player's velocity. The controlling team is that player's team. |

There are 12 features:

1. Centred rolling means of home control and away control over ±12, ±50 and ±150 frames (6 features).
2. Time since the last and until the next home control and away control, capped at 60 s (4 features).
3. The last and next controlling team, encoded +1 home, −1 away, 0 none (2 features).

All windows stay within a half.

## Variants and protocol

- **Reference C:** out-of-fold v4 recipe, the stored `external/oof/raw` probabilities.
- **Candidate D:** C's 60 features plus the 12 control features, with the same HistGradientBoosting settings and the same training rule. Data, folds, conditions and seeds are identical to EXPERIMENT-06.
- **Decoder:** the v4 decoder unchanged (w 1, uniform penalty 0.5).
- **Scoring:** the official evaluator under the four EXPERIMENT-08 conventions.

## Gate (D vs C, out of fold)

All of the following must hold:

- SkillCorner convention-averaged mean spellF1 delta ≥ +0.015;
- wins on at least 14 of 20 SkillCorner matches;
- DFL contested convention-averaged mean delta ≥ −0.005;
- DFL natural ≥ −0.02;
- SkillCorner raw macroF1 delta ≥ −0.02.

## If the gate passes

1. Retrain D on all 27 matches and freeze it with hashes.
2. Add a verified inference entry and run the stability checks.
3. Run two identical Mac runs.
4. Run the label-free EXPERIMENT-07 tag agreement, which uses no tags in the model and so is not circular. It must not fall more than 0.02 below v4's 0.79.

## Limits

Cross-validation on spent data; provider label conventions; 25 fps upsampling for SkillCorner.
