# Ball possession from automatic tracking: technical note

More Than A Hack 2027 Phase 1 submission. Team: **Mastaadmi**. Possession file: `Mastaadmi.csv` (version v10). Code: https://github.com/nilaymastaadmi/mastaadmi-mtah-2027 (MIT licence).

## Summary

We label each of the 199,319 tactical-video frames as H, A, D or X with one fixed algorithm. It reads only the two tracking feeds in the data bundle (tactical and broadcast). A model trained on other matches estimates, frame by frame, which team has the ball or whether play is dead, and a decoder turns those estimates into whole possessions. No frame of the provided match was labelled by hand, and no model was trained, fitted or tuned on it.

What we can and cannot claim: on 27 other matches the method clearly beats the organisers' baseline and our earlier versions. On the competition match we have no ground truth, so we do not know its score. The last two changes were small on real tracking: borrowing the ball position from the second tracking feed (v8) and replacing the per-frame model by a temporal network (v10). Section 4 gives their sizes, and Section 6 the caveats on the v10 number.

## 1. Approach

The pipeline has four steps.

**Step 0: fill missing ball positions from the second feed.** The tactical tracking has no ball position on 41% of graded frames, and the ball is the strongest cue for possession. The bundle also holds a broadcast-view tracking feed of the same match. We align it to the tactical timeline with no identities, no clock and no labels:

1. try the four mirror orientations and keep the best;
2. find one constant frame offset per half by voting on how well the players of the two feeds line up;
3. split the broadcast feed into camera shots and re-check each shot locally, so replays and graphics are rejected;
4. accept a frame only if its players sit within 1.2 m of the tactical players on average.

Where the tactical ball is missing and an accepted broadcast frame has one, the broadcast ball is used. A tactical ball is never overwritten.

**Step 1: per-frame observations (90 columns).** None uses player identities, clock time or club information.

- *Ball-centric (8):* nearest home and away player distance, ball speed, stationary-ball flag, length of the current missing-ball gap, outside-pitch flag, visibility, mean player speed.
- *Temporal context (29):* home-minus-away nearest-distance difference, distance to the nearest line, per-team counts within 5 m and per-team speeds, time since and until the ball is visible, last and next visible ball context, and rolling means over ±0.5 s, 2 s and 5 s.
- *Team shape (23):* for each team in its own attacking direction: centroid depth, spread, length, width, back-line depth, player count and advance per second, plus home-minus-away differences. The attacking direction of each half is inferred from the team centroids. The team with the ball spreads out and pushes up, and the other team compacts in front of its goal. This stays available while the ball is missing.
- *Ball control (12):* a frame counts as controlled when the ball is visible, within 1.5 m of the nearest player, and moving within 2.5 m/s of that player's velocity. From that we build rolling control shares per team, time since and until each team's last and next control, and the last and next controlling team. This follows the competition's definition of possession as control, not proximity.
- *Ball geometry (18, added in v10):* ball visible flag, ball speed, change of ball speed and direction between 3 frames before and after, for each team the nearest and second-nearest player distance and the speed of the nearest player relative to the ball, players within 5 m, which team is nearest and whether that changed in the last 5 frames, and the ball's distance to the touchline and goal line.

**Step 2: learned evidence (v10).** A dilated temporal convolution network reads the 90 columns of a whole half at once. A 1×1 convolution to 64 channels is followed by 7 residual blocks (layer norm, kernel-3 dilated convolution with dilations 1, 2, 4, 8, 16, 32 and 48, GELU, 1×1 convolution, dropout 0.1). Each frame therefore sees about ±4.4 s of context, and the model can tell a short turnover from a touch by looking before and after it. It outputs P(H, A, D) per frame, and a second head for "possession changes here". The second head is not used by the final decoder (Section 2). Training: 12 epochs of AdamW on random 2,048-frame chunks, fixed seed, CPU, deterministic algorithms. The model was trained only on other matches (Section 3). Its weights are in the repository (`v10/final.pt`, sha256 5fe16674…1bee).

**Step 3: decoding.** A three-state Viterbi decoder runs per half on −log P / 25 and returns the single most coherent H/A/D sequence. Switching between the two teams costs 1.0, and entering or leaving dead ball costs 0.5. Frames outside the two halves are X. (Before v10 every switch cost 0.5 and the evidence came from a per-frame gradient-boosting model with 72 of these columns.)

**Why this design.** The ranking metric (spellF1) rewards whole possessions. An oracle test on other matches (the bench, Section 6) showed that v8 lost most of its score by missing short possessions, which were absorbed into a long possession of the other team or merged into one of the same team. A decoder search alone had gained only +0.005, so the v10 experiment gave the model new evidence about turnovers instead, by letting it see the frames before and after each moment.

**Compliance with the 4 November clarification.** Every model was trained, and every parameter chosen, on external matches only. The provided tracking, the Smart Tagging file and the video were used as inputs or for automatic checks. The ground truth, the public five-minute labels and any hand labelling were never used to train, tune or choose a version.

## 2. Assets used and dropped

| Asset | What happened |
|---|---|
| Tactical tracking (`BAR-ATH_tactical_atd.txt`, `_atd.xml`) | **Used.** Main input to every step. |
| Broadcast tracking (`BAR-ATH_broadcast_atd.*`) | **Used (EXPERIMENT-13, v8).** Only to fill missing ball frames (Step 0). Its players and ball are never fed to the model on their own. |
| Smart Tagging (`_smart_tagging.xml`) | **Tried as model input, dropped (EXPERIMENT-09).** Fusing the tagged team into the decoder never helped, even when we simulated tags that were 95% correct, and it hurt at strong weights. **Kept as an automatic check:** agreement between our team calls and the tagged possession team over build-up, progression, transitions and long balls. Agreement was 0.687 (v2), 0.792 (v4), 0.777 (v6), 0.783 (v8) and 0.772 (v10). A new version had to stay within 0.02 of the previous one; the check was never used to rank candidates. |
| Tactical video | **Tried, dropped (EXPERIMENT-17).** We ran a pretrained ball detector (WASB, soccer weights trained by its authors on the ISSIA-CNR dataset) over every frame to fill missing balls. It passed our screen on ISSIA, but on the match its balls were a median 2.24 m from the tracked ball where both existed (limit 1.0 m), 46.5% were more than 3 m apart, and it filled only 1.7% of graded frames. The registered stop rule was met, so no video-derived value is in the submission. We trained and fine-tuned nothing. Details: `outputs/experiment-17-results.md`. |
| Homography file | Used only inside the dropped video experiment. |
| Public five-minute labels | Used once, for an early version, as a pipeline check (to catch a home/away inversion). Never for training or tuning, and never to choose between versions. |

Tried and dropped on external matches, each with a pass/fail gate fixed before the run:

- **Possession-change detector (EXPERIMENT-18, arm B).** A gradient-boosting model that flags frames where possession changes hands, feeding the decoder as a cheaper cost for switching there. It ranks change frames well (AUC 0.87 on real tracking) but barely moved the score: +0.002 on real tracking, 13 of 20 matches, against +0.02 required. The temporal network's own change head and a blend of the two (arm AB, +0.023) were not preferred, and the final decoder ignores change evidence (the selected weight was 0 in every fold).
- Temporal stacking on the earlier model's probabilities (EXPERIMENT-11), a blend of two versions (14), larger and seed-averaged gradient-boosting models (12) and interpolation over short ball gaps (15, there are no short gaps).

## 3. Training data (none from the provided match)

| Dataset | Use | Who labelled, and how | Amount |
|---|---|---|---|
| IDSSE/DFL open data: Bassek, Rein, Weber & Memmert (2025), *Scientific Data* 12:195, doi 10.6084/m9.figshare.28196177.v1, CC BY 4.0 | training and validation | The data provider's per-frame BallStatus and BallPossession flags: dead → D, home → H, guest → A. We did not label anything. | 7 Bundesliga matches, 25 fps |
| SkillCorner open data, github.com/SkillCorner/opendata @ 4340d27, MIT | training and validation on real broadcast-tracking noise | Per-frame team possession produced by SkillCorner's own tracking pipeline; no possession → D. | 20 A-League matches, 10 fps upsampled to 25 fps |
| Metrica Sports sample data, github.com/metrica-sports/sample-data @ e706dd5 | earlier versions and diagnostics only | Event-derived proxy. The submitted model does not use it. | 3 matches |

The submitted model (v10) was trained on the 27 DFL and SkillCorner matches together, with the same recipe that was validated out of fold.

DFL matches were also used as deterministic degraded copies, so that the model sees input as bad as the competition's:

- ball removed in long runs (41% of frames; median gap 10.7 s, p90 34 s, maximum 221 s);
- position jitter, false ball positions and player dropout.

These settings were calibrated only to aggregate ball-visibility statistics of the provided tracking (the 41% figure is also published by the organisers). SkillCorner frames count in the training loss only where the ball is visible, because in broadcast video a missing ball mostly signals a stoppage, which would not transfer to the tactical camera. They are still fed to the network as context.

The fusion step has no learned parameters. Its constants (1.2 m acceptance, shot length limits, search window) were fixed on simulated second views of the DFL matches before any result was seen. We disclose one deviation: the simulated view used 0.15 m position noise where the registration said 0.3 m; a later run at 0.3 m (EXPERIMENT-16) gave the same conclusion.

The ISSIA-CNR data (non-commercial research terms) was used only as a held-out test of the third-party video detector in the dropped experiment. Nothing from it is in the submission or the public repository.

## 4. Validation

Every experiment was registered (data, split, candidates, selection rule, gates) before its outcome, and scored with the official `possession_metrics.evaluate` on full matches. These are results on other matches, not estimates of the competition score, and their absolute values are not comparable to the competition board.

**The model, out of fold.** Match-grouped 5-fold cross-validation over the 27 matches: each match was scored by a model that never saw it. spellF1 under the data providers' own labels:

| spellF1, provider labels | 20 real broadcast matches | 7 DFL matches, simulated gaps | 7 DFL matches, clean |
|---|---:|---:|---:|
| **Submitted model (v10)** | **0.486** | **0.545** | **0.665** |
| v8 / v6 model (per-frame gradient boosting) | 0.460 | 0.426 | 0.544 |
| v4 (without ball control) | 0.444 | 0.414 | 0.494 |
| v2 (first learned version) | 0.311 | 0.298 | 0.507 |
| Organisers' baseline ±50 | 0.249 | 0.177 | 0.308 |

Provider labels count deflections and touches that the competition does not, so we also average over four label conventions (provider labels, and controlled-possession rules at 1, 2 and 3 s). On that average v10 against v8 is:

| mean of 4 conventions | v8 | v10 | change | matches won |
|---|---:|---:|---:|---|
| 20 real broadcast matches | 0.512 | 0.538 | +0.026 | 20 of 20 |
| 7 DFL, simulated gaps | 0.494 | 0.600 | +0.106 | 7 of 7 |
| 7 DFL, clean | 0.627 | 0.716 | +0.089 | 7 of 7 |

Under the strictest rule (3 s controlled possession) the change is +0.027 on real tracking and +0.073 on gaps; real-tracking macroF1 rises from 0.855 to 0.866. The gate fixed in advance (+0.02 on real tracking and on gaps, a win on at least 15 of 20 real matches, and guards on clean DFL and macroF1) was passed on all 7 checks (`outputs/experiment-18-results.md`).

**How to read the v10 gain.** On real tracking, which is what the competition match is, the gain is +0.026, a quarter of the DFL gain. It comes mostly from cleaner segmentation, not from finding many more short possessions: predicted possessions that sit over true dead time fall from 9.2 to 5.0 per match and over the other team from 26.2 to 23.8, while recall of true possessions under 3 s only moves from 11% to 12% (raw labels). On DFL the model does find short possessions (recall under 3 s from 17% to 31% on clean data), and that is where the large gains come from. Part of the DFL gaps gain may be the model learning our own gap simulation. One real match is a near tie (+0.0002). Further, the cost of switching between teams rose from 0.5 to 1.0 in the same step. Against v8 decoded at its own best switch cost (1.0) the real-tracking gain is +0.022, with 17 of 20 matches won and the worst match at −0.001; the temporal model at v8's cost (0.5) gains +0.014. So a realistic change in the private score is a few hundredths, not the DFL-sized +0.1.

**The fusion (v8 input).** On a simulated second view of the 7 DFL matches with gaps, filling the ball raised spellF1 by +0.043 on average (7 of 7 matches, macroF1 +0.013) and cut missing ball from 41.5% to 21.4% (`outputs/experiment-13-results.md`). With noisier simulated views the gain shrinks: +0.033 at 0.3 m and +0.018 at 0.5 m, still 7 of 7 (`outputs/experiment-16-results.md`). The real broadcast feed is less helpful than the simulation. Label-free checks on the competition match: one clean offset per half; 96.2% of aligned frames within one frame of it; player residual median 0.49 m, p90 0.91 m; ball agreement with the tactical ball median 0.73 m over 65,574 frames where both see it. But only 4,936 missing-ball frames were recovered: missing ball fell from 40.7% to 37.6% of graded frames, because the two cameras mostly lose the ball at the same moments. We estimate a real gain of about +0.005 from this step. That is our estimate, not a measurement.

**v10 on the competition match (label-free).** Two full runs were byte-identical and the official validator passed. Agreement with Smart Tagging was 0.772 against a registered floor of 0.763 (v8 0.783, v6 0.777). The file differs from v8 on 8.1% of graded frames, with about 239 possessions against 255 and a dead-ball share of 0.484 against 0.475. We adopted v10 because it met the rules fixed in advance.

**Caveats on the evidence.**

- **Nested selection (disclosed, not fixed).** The decoder setting was chosen fold by fold, keeping each fold's own labels out of its choice. But the scores used came from models that had themselves trained on that fold, so its labels reach the choice indirectly. A strict version would refit every model without that fold, which we did not do. All five folds chose the same setting (team-switch cost 1.0, change weight 0), so any inflation is limited to that one choice, and we have not measured its size.
- **Several candidates, one gate.** Three variants (A, B and AB) were tested against one gate. We kept the thresholds for that reason; the winner was the single registered pick.
- **No untouched holdout.** Every public tracking dataset with possession labels that we know of has been used in earlier tests, so this is cross-validation. The first learned version (v2) was tested one-shot on 4 untouched DFL matches and then on 19 untouched real broadcast matches, and beat the baseline and the best tuned non-learning method on every match; later versions were selected by cross-validation on the same pool.

## 5. Possession rules as implemented

- **Who has the ball:** learned from ball proximity, explicit control (ball moving with a player), ball geometry, temporal context over about ±4.4 s and team shape. A switch between teams needs sustained evidence (cost 1.0), so touches, deflections and short aerial contests usually do not flip the label. This approximates "controlled possession".
- **Loose balls, challenges and missing ball:** the decoder keeps the current team unless the evidence, including team shape, pays for a switch. Contested balls stay with the last team. A missing ball alone is never read as dead.
- **Dead ball:** learned from the providers' ball-status labels. Cues include a stationary ball, a ball near or outside the lines, and play-wide low speed.
- **Restarts:** handled implicitly as D-to-team transitions. No rule is tied to specific frames or events of the provided match.
- **Abstention:** X only outside the two graded halves.

## 6. Limitations

The first bullets are from the bench on v8 (`outputs/bench-loss-decomposition.md`) and the v10 results (`outputs/experiment-18-results.md`). All are from other matches, not from the competition match.

- **We still find too few possessions on real tracking.** v8 predicted about 215 possessions per SkillCorner match against 290 true ones (235 against 358 on DFL), with mean lengths of 13 to 15 s against 9 to 11 s. v10 predicts 211 on real tracking and 287 on clean DFL. True possessions under 3 s are matched only 12% of the time on real tracking (raw labels; 19% under the 3 s rule). Resolving the merged possessions alone would be worth up to +0.10 to +0.11 under the 3 s rule and more under provider labels (v8 numbers). v10 closes only a small part of that on real tracking.
- **The v10 gain is small on real tracking and larger on DFL** (Section 4). The nested-selection and gap-simulation caveats apply.
- **Missing ball still costs about 0.10 spellF1** on the simulated gaps. The second feed recovers little on the real match, and the video detector recovers almost nothing, because the frames the tactical tracking misses are the ones other views miss too.
- **Dead ball:** a perfect dead-ball mask would add +0.08 on real tracking. v8 missed about 19 of 112 stoppages per match and invented about 14. We did not re-measure this for v10.
- **Label conventions differ.** DFL and SkillCorner define possession and dead ball in their own way, and the model is trained on provider labels that count deflections. They give mean possessions of 7 to 12 s and 45% to 47% dead time. The organisers' published statistics for the five-minute window (mean possession 14.2 s, 19.6% dead) are different, and one window cannot say which convention the full ground truth follows. We did not tune to it.
- **Cross-machine reproducibility is exact for the labels, not for the probabilities.** On a Linux x86 machine with CPU torch the decoded labels were identical on 17 of 17 public views (0 of 2,454,886 frames differ), but the network's probabilities differ by up to 9.5e-7 and the input features differ in their last digits. A near-tie in the decoder could therefore flip a frame on another machine; none did on 2.45 million public frames. The digest below is for macOS arm64.
- **Domain:** Bundesliga and A-League training data, synthetic degradation for DFL, SkillCorner upsampled from 10 fps.
- **No untouched final test and no competition-side score.** We make no claim about our rank.

With more time we would (1) refit every model without each held-out fold for a strictly nested selection, (2) find dead-ball evidence that does not depend on the ball, and (3) test on a new independent dataset.

## 7. Reproduction

You need Python 3.13 (we froze with 3.13.16) and the organisers' bundle placed as:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/broadcast/BAR-ATH_broadcast_atd.txt
data/broadcast/BAR-ATH_broadcast_atd.xml
```

There are two environments. `.venv` holds the pinned `requirements.txt` (numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1) and runs the alignment, the features and the CSV writer. `.venv-video` holds `requirements-v10.txt` (torch 2.15.0, numpy 2.5.4) and runs the network (the name is historical). On Linux, install the CPU wheel with `--index-url https://download.pytorch.org/whl/cpu`.

```sh
python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
python3.13 -m venv .venv-video && .venv-video/bin/python -m pip install -r requirements-v10.txt
OMP_NUM_THREADS=2 .venv/bin/python -m unittest test_possession test_first_assessment v2.test_v2 v3.test_v3 v4.test_v4 v6.test_v6 v8.test_v8 exp18.test_exp18 exp18.test_arm_b
bash v10/run.sh data
```

`v10/run.sh` runs three stages twice and compares the two CSVs. Stage 1 (`v10.inputs`) aligns the broadcast feed, fills the ball and builds the 90 input columns. Stage 2 (`v10.predict`) runs the frozen network and decoder. Stage 3 (`v10.write`) writes the CSV and validates it with the organisers' `validate_submission.py`. Stage 2 refuses to run unless the recorded gate passed and the model and every bound source file (35 in all) match their frozen hashes. Output goes to `work/v10-evaluation/`; copy `v10_<stamp>_run1.csv` to `Mastaadmi.csv`.

**Final file SHA-256:** `51bc7d93f1c32dbd6a65015a9a1451f9abedb66dce3d67f807036dc3b06eadcb`. Generated twice on macOS arm64 with Python 3.13.16, identical both times; the official validator passed. The Linux check (`work/v10-evaluation/portability-compare.json`) ran the same frozen model and decoder on a Linux x86 runner (public matches only, no competition data) and found identical labels (Section 6). `python -m v10.portability` is the script.

Earlier versions remain as fallbacks and reproduce with their own commands: v8 (`v8.infer`) `bb0ba6f55369f862a6ea2151882b82691a7331d6d98733bf6fe6225d08c4034e`; v6 (`v6.infer`) `98d5f2b29d1f1707656da82f3f214ac7e01a99f2714c22f2fbe02c051160ed91`; v4 (`v4.infer`) `fd8d9430baa68e7d84cb42ef17bdd4cb58b99cca771454151156433c84518837`; v2 (`v2.verified_inference`) `c69eeb177b2d13c35b3235d4226aa1e0827e339c517a9bd4629e53b11535d481`.

## Credits

IDSSE/DFL data (CC BY 4.0), SkillCorner open data (MIT), Metrica Sports sample data and the organisers' starter code (MIT) are credited in `THIRD_PARTY.md`. The WASB detector (MIT code) and ISSIA-CNR data were used only in the dropped video experiment.
