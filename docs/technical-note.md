# Ball possession from automatic tracking: technical note

More Than A Hack 2027 Phase 1 submission. Team: **Mastaadmi**. Possession file: `Mastaadmi.csv` (version v8). Code: https://github.com/nilaymastaadmi/mastaadmi-mtah-2027 (MIT licence).

## Summary

We label each of the 199,319 tactical-video frames as H, A, D or X with one fixed algorithm. It reads only the two tracking feeds in the data bundle (tactical and broadcast). A model trained on other matches estimates, frame by frame, which team has the ball or whether play is dead, and a decoder turns those estimates into whole possessions. No frame of the provided match was labelled by hand, and no model was trained, fitted or tuned on it.

What we can and cannot claim: on 27 other matches the method clearly beats the organisers' baseline and our earlier versions. On the competition match we have no ground truth, so we do not know its score. The last change (v8, borrowing the ball position from the second tracking feed) is small on this match, and we say how small in Section 4.

## 1. Approach

The pipeline has four steps.

**Step 0: fill missing ball positions from the second feed (v8).** The tactical tracking has no ball position on 41% of graded frames, and the ball is the strongest cue for possession. The bundle also holds a broadcast-view tracking feed of the same match. We align it to the tactical timeline with no identities, no clock and no labels:

1. try the four mirror orientations and keep the best;
2. find one constant frame offset per half by voting on how well the players of the two feeds line up;
3. split the broadcast feed into camera shots and re-check each shot locally, so replays and graphics are rejected;
4. accept a frame only if its players sit within 1.2 m of the tactical players on average.

Where the tactical ball is missing and an accepted broadcast frame has one, the broadcast ball is used. A tactical ball is never overwritten.

**Step 1: per-frame observations (72 features).** None uses player identities, clock time or club information.

- *Ball-centric (8):* nearest home and away player distance, ball speed, stationary-ball flag, length of the current missing-ball gap, outside-pitch flag, visibility, mean player speed.
- *Temporal context (29):* home-minus-away nearest-distance difference, distance to the nearest line, per-team counts within 5 m and per-team speeds, time since and until the ball is visible, last and next visible ball context, and rolling means over ±0.5 s, 2 s and 5 s.
- *Team shape (23):* for each team in its own attacking direction: centroid depth, spread, length, width, back-line depth, player count and advance per second, plus home-minus-away differences. The attacking direction of each half is inferred from the team centroids. The team with the ball spreads out and pushes up, and the other team compacts in front of its goal. This stays available while the ball is missing.
- *Ball control (12):* a frame counts as controlled when the ball is visible, within 1.5 m of the nearest player, and moving within 2.5 m/s of that player's velocity. From that we build rolling control shares per team over ±0.5 s, 2 s and 6 s, time since and until each team's last and next control, and the last and next controlling team. This follows the competition's definition of possession as control, not proximity.

**Step 2: learned evidence.** A histogram gradient-boosting classifier (scikit-learn; 100 iterations, 15 leaves, fixed seed) estimates P(H, A, D | features). It was trained only on other matches (Section 3).

**Step 3: decoding.** A three-state Viterbi decoder runs per half on −log P / 25 with a uniform switch penalty of 0.5 and returns the single most coherent H/A/D sequence. Frames outside the two halves are X.

**Why this design.** The ranking metric (spellF1) rewards whole possessions, so the decoder suppresses flicker. An oracle test on other matches located the main error: giving the model the true team during live play lifted spellF1 from 0.32 to 0.70, far more than perfect dead-ball timing did (0.45). That is why we added team shape (which works without the ball) and ball control, and why we looked for more ball positions (Step 0).

**Compliance with the 4 November clarification.** Every model was trained, and every parameter chosen, on external matches only. The provided tracking, the Smart Tagging file and the video were used as inputs or for automatic checks. The ground truth, the public five-minute labels and any hand labelling were never used to train, tune or choose a version.

## 2. Assets used and dropped

| Asset | What happened |
|---|---|
| Tactical tracking (`BAR-ATH_tactical_atd.txt`, `_atd.xml`) | **Used.** Main input to every step. |
| Broadcast tracking (`BAR-ATH_broadcast_atd.*`) | **Used (EXPERIMENT-13, v8).** Only to fill missing ball frames (Step 0). Its players and ball are never fed to the model on their own. |
| Smart Tagging (`_smart_tagging.xml`) | **Tried as model input, dropped (EXPERIMENT-09).** Fusing the tagged team into the decoder never helped, even when we simulated tags that were 95% correct, and it hurt at strong weights. **Kept as an automatic check:** agreement between our team calls and the tagged possession team over build-up, progression, transitions and long balls. Agreement was 0.687 (v2), 0.792 (v4), 0.777 (v6) and 0.783 (v8). A new version had to stay within 0.02 of the previous one; the check was never used to rank candidates. |
| Tactical video | **Tried, dropped (EXPERIMENT-17).** We ran a pretrained ball detector (WASB, soccer weights trained by its authors on the ISSIA-CNR dataset) over every frame to fill missing balls. It passed our screen on ISSIA, but on the match its balls were a median 2.24 m from the tracked ball where both existed (limit 1.0 m), 46.5% were more than 3 m apart, and it filled only 1.7% of graded frames. The registered stop rule was met, so no video-derived value is in the submission. We trained and fine-tuned nothing. Details: `outputs/experiment-17-results.md`. |
| Homography file | Used only inside the dropped video experiment. |
| Public five-minute labels | Used once, for an early version, as a pipeline check (to catch a home/away inversion). Never for training or tuning, and never to choose between versions. |

We also tried and dropped, on external matches: temporal stacking on the model's probabilities (EXPERIMENT-11), a blend of two versions (14), larger and seed-averaged models (12) and interpolation over short ball gaps (15, there are no short gaps). Each had a pass/fail gate fixed before the run.

## 3. Training data (none from the provided match)

| Dataset | Use | Who labelled, and how | Amount |
|---|---|---|---|
| IDSSE/DFL open data: Bassek, Rein, Weber & Memmert (2025), *Scientific Data* 12:195, doi 10.6084/m9.figshare.28196177.v1, CC BY 4.0 | training and validation | The data provider's per-frame BallStatus and BallPossession flags: dead → D, home → H, guest → A. We did not label anything. | 7 Bundesliga matches, 25 fps |
| SkillCorner open data, github.com/SkillCorner/opendata @ 4340d27, MIT | training and validation on real broadcast-tracking noise | Per-frame team possession produced by SkillCorner's own tracking pipeline; no possession → D. | 20 A-League matches, 10 fps upsampled to 25 fps |
| Metrica Sports sample data, github.com/metrica-sports/sample-data @ e706dd5 | earlier versions and diagnostics only | Event-derived proxy. The submitted model does not use it. | 3 matches |

The submitted model was trained on the 27 DFL and SkillCorner matches together: 4,762,447 frames.

DFL matches were also used as deterministic degraded copies, so that the model sees input as bad as the competition's:

- ball removed in long runs (41% of frames; median gap 10.7 s, p90 34 s, maximum 221 s);
- position jitter, false ball positions and player dropout.

These settings were calibrated only to aggregate ball-visibility statistics of the provided tracking (the 41% figure is also published by the organisers). SkillCorner frames were used for training only where the ball is visible, because in broadcast video a missing ball mostly signals a stoppage, which would not transfer to the tactical camera.

The fusion step has no learned parameters. Its constants (1.2 m acceptance, shot length limits, search window) were fixed on simulated second views of the DFL matches before any result was seen. We disclose one deviation: the simulated view used 0.15 m position noise where the registration said 0.3 m; a later run at 0.3 m (EXPERIMENT-16) gave the same conclusion.

The ISSIA-CNR data (non-commercial research terms) was used only as a held-out test of the third-party video detector in the dropped experiment. Nothing from it is in the submission or the public repository.

## 4. Validation

Every experiment was registered (data, split, candidates, selection rule, gates) before its outcome, and scored with the official `possession_metrics.evaluate` on full matches. These are results on other matches, not estimates of the competition score, and their absolute values are not comparable to the competition board.

**The model (v6 recipe, unchanged inside v8).** Match-grouped 5-fold cross-validation over the 27 matches: each match was scored by a model that never saw it. spellF1 under the data providers' own labels:

| spellF1, provider labels | 20 real broadcast matches | 7 DFL matches, simulated gaps | 7 DFL matches, clean |
|---|---:|---:|---:|
| **Submitted model (v6 recipe)** | **0.460** | **0.426** | **0.544** |
| v4 (without ball control) | 0.444 | 0.414 | 0.494 |
| v2 (first learned version) | 0.311 | 0.298 | 0.507 |
| Organisers' baseline ±50 | 0.249 | 0.177 | 0.308 |

Provider labels count deflections and touches that the competition does not, so the same predictions also score 0.512, 0.494 and 0.627 when averaged over four label conventions (provider labels, and controlled-possession rules at 1, 2 and 3 s). Real-tracking macroF1 is 0.855. Ball control (v4 to v6) beat v4 on 16 of 20 real matches (+0.015, a thin margin) and on 7 of 7 clean DFL matches (+0.047). The v4 and v2 rows come from EXPERIMENT-06 and the submitted row from the bench (`outputs/bench-loss-decomposition.md`, section 1).

**The fusion (v8).** On a simulated second view of the 7 DFL matches with gaps, filling the ball raised spellF1 by +0.043 on average (range +0.028 to +0.052, 7 of 7 matches, macroF1 +0.013) and cut missing ball from 41.5% to 21.4% (`outputs/experiment-13-results.md`). With noisier simulated views the gain shrinks: +0.033 at 0.3 m and +0.018 at 0.5 m, still 7 of 7 (`outputs/experiment-16-results.md`).

The real broadcast feed is less helpful than the simulation. Label-free checks on the competition match:

- one clean offset per half; 96.2% of aligned frames within one frame of it;
- player residual median 0.49 m, p90 0.91 m;
- ball agreement with the tactical ball: median 0.73 m over 65,574 frames where both see it;
- but only 4,936 missing-ball frames were recovered: missing ball fell from 40.7% to 37.6% of graded frames, because the two cameras mostly lose the ball at the same moments.

From that recovery rate we expect a real gain of about +0.005 spellF1. That is our estimate, not a measurement: without ground truth we cannot score this match, and the gain is inside the noise of our own validation. We adopted v8 because it met the rules fixed in advance: instrument checks passed, two runs byte-identical, official validator passed, zero decoded-frame changes under perturbation, and tag agreement 0.783 against 0.777 for v6.

**Caveat.** Every public tracking dataset with possession labels that we know of has been used in earlier tests, so the evidence is cross-validation, not an untouched holdout. The first learned version (v2) was tested one-shot on 4 untouched DFL matches and then on 19 untouched real broadcast matches, and beat the baseline and the best tuned non-learning method on every match; later versions were selected by cross-validation on the same pool.

## 5. Possession rules as implemented

- **Who has the ball:** learned from ball proximity, explicit control (ball moving with a player), temporal context and team shape. A switch needs sustained evidence (penalty 0.5), so touches, deflections and short aerial contests usually do not flip the label. This approximates "controlled possession".
- **Loose balls, challenges and missing ball:** the decoder keeps the current team unless the evidence, including team shape, pays for a switch. Contested balls stay with the last team. A missing ball alone is never read as dead.
- **Dead ball:** learned from the providers' ball-status labels. Cues include a stationary ball, a ball near or outside the lines, and play-wide low speed.
- **Restarts:** handled implicitly as D-to-team transitions. No rule is tied to specific frames or events of the provided match.
- **Abstention:** X only outside the two graded halves.

## 6. Limitations

All numbers below are from other matches (`outputs/bench-loss-decomposition.md`), not from the competition match.

- **We find too few possessions.** The model predicts about 215 possessions per SkillCorner match against 290 true ones (235 against 358 on DFL). Its possessions average 13 to 15 s; the true ones 9 to 11 s. Only 52% to 60% of true possessions are matched.
- **The misses are the short ones.** True possessions under 3 s are matched only 11% to 17% of the time (provider labels). Most are absorbed by a long correct prediction for the other team, or merged into a longer one for the same team. Resolving them is worth up to +0.29 to +0.36 spellF1 under provider labels, and +0.10 to +0.11 under the 3 s controlled rule. This is the biggest known lever, and we did not solve it.
- **Missing ball still costs about 0.10 spellF1** on the simulated gaps. The second feed recovers little on the real match, and the video detector recovers almost nothing, because the frames the tactical tracking misses are the ones other views miss too.
- **Dead ball:** a perfect dead-ball mask would add +0.08 on real tracking. The model misses about 19 of 112 stoppages per match and invents about 14.
- **Label conventions differ.** DFL and SkillCorner define possession and dead ball in their own way. They give mean possessions of 7 to 12 s and 45% to 47% dead time. The organisers' published statistics for the five-minute window (mean possession 14.2 s, 19.6% dead) are different, and one window cannot say which convention the full ground truth follows. We did not tune to it.
- **Domain:** Bundesliga and A-League training data, synthetic degradation for DFL, SkillCorner upsampled from 10 fps.
- **No untouched final test and no competition-side score.** We make no claim about our rank.

With more time we would (1) build a detector for short turnovers to fix the under-segmentation, (2) find dead-ball evidence that does not depend on the ball, and (3) test on a new independent dataset.

## 7. Reproduction

You need Python 3.13 (we froze with 3.13.16), the pinned `requirements.txt` (numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1), and the organisers' bundle placed as:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/broadcast/BAR-ATH_broadcast_atd.txt
data/broadcast/BAR-ATH_broadcast_atd.xml
```

```sh
python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 .venv/bin/python -m unittest test_possession test_first_assessment v2.test_v2 v3.test_v3 v4.test_v4 v6.test_v6 v8.test_v8
OMP_NUM_THREADS=1 .venv/bin/python -m v8.infer --data-dir data --out Mastaadmi.csv
```

`v8.infer` refuses to run unless the pre-registered gate passed and the frozen v6 model, its settings, every bound source file and the package versions match their recorded hashes. It then writes the CSV, validates it with the organisers' `validate_submission.py`, and prints the SHA-256. Alignment uses all CPU cores; it took about 10 minutes on a busy laptop.

`python -m v8.stability_check --compare` (needs the public DFL files, see `README.md`) shows that perturbations far larger than cross-platform floating-point differences change no alignment or decoded frame, and that the outputs match digests stored from a Linux run.

**Final file SHA-256:** `bb0ba6f55369f862a6ea2151882b82691a7331d6d98733bf6fe6225d08c4034e`. Generated twice on macOS arm64 with Python 3.13.16, identical both times; the official validator passed; zero changes under perturbation of the competition inputs.

Fallbacks, all reproduced the same way: v6 (`v6.infer`) `98d5f2b29d1f1707656da82f3f214ac7e01a99f2714c22f2fbe02c051160ed91`; v4 (`v4.infer`) `fd8d9430baa68e7d84cb42ef17bdd4cb58b99cca771454151156433c84518837`; v2 (`v2.verified_inference`) `c69eeb177b2d13c35b3235d4226aa1e0827e339c517a9bd4629e53b11535d481`.

## Credits

IDSSE/DFL data (CC BY 4.0), SkillCorner open data (MIT), Metrica Sports sample data and the organisers' starter code (MIT) are credited in `THIRD_PARTY.md`. The WASB detector (MIT code) and ISSIA-CNR data were used only in the dropped video experiment.
