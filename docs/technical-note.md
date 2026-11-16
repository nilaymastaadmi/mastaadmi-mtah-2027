# Ball possession from automatic tracking: technical note

More Than A Hack 2027 Phase 1 submission. Team: **Mastaadmi**. Possession file: `Mastaadmi.csv` (version v12). Code: https://github.com/nilaymastaadmi/mastaadmi-mtah-2027 (MIT licence).

## Summary

We label each of the 199,319 tactical-video frames as H, A, D or X with one fixed algorithm. It reads only the two tracking feeds in the data bundle (tactical and broadcast). Two models trained on other matches estimate, frame by frame, which team has the ball or whether play is dead: a gradient-boosted model with a few seconds of context (v8) and a temporal convolutional network with about 25 s of context each side (v12). We average their probabilities, and a decoder turns the average into whole possessions. No frame of the provided match was labelled by hand, and no model was trained, fitted or tuned on it.

What we can and cannot claim: on 64 PFF World Cup matches untouched by training, v12 beats our previous submission (v8) by +0.030 spellF1 (mean of four label conventions) and wins 62 of the 64 matches. Out of fold it beats v8 on 20 SkillCorner matches (+0.015, 18 of 20) and on 7 DFL matches with simulated ball gaps (+0.118, 7 of 7). On the competition match we have no ground truth beyond the public five-minute window, so we do not know its full score. There v12 predicts 227 possession starts against 253 for v8, the opposite direction from PFF (Section 6).

History in short: v1 to v6 added learned per-frame models, team shape and ball control. v8 added ball positions from the second feed. v10, a temporal network, beat v8 in cross-validation, lost on the 64 PFF matches, and we withdrew it. v12 passed a gate fixed before any v12 score, and we adopted it on 16 November 2026.

## 1. Approach

Steps 0, 1, 2a and 3 are v8 and are unchanged in v12. Step 2b is new.

**Step 0: fill missing ball positions from the second feed (v8).** The tactical tracking has no ball position on 41% of graded frames, and the ball is the strongest cue for possession. We align the broadcast feed to the tactical timeline from player positions alone (no identities, clock or labels): best of four mirror orientations, one frame offset per half, a local re-check per camera shot so replays are rejected, and a frame is accepted only if its players sit within 1.2 m of the tactical players on average. Where the tactical ball is missing and an accepted broadcast frame has one, the broadcast ball is used. A tactical ball is never overwritten.

**Step 1: per-frame observations.** None uses player identities, clock time or club information. v8 reads 72 features; the v12 network reads 90 columns (the same 72 plus 18 geometry columns), plus a missing-value flag for each of the 57 columns that can be missing (147 inputs).

- *Ball-centric (8) and temporal context (29):* nearest player distances, ball speed, stationary and outside-pitch flags, missing-ball gap length, per-team counts within 5 m, and rolling means over 0.5 s, 2 s and 5 s each side.
- *Team shape (23):* for each team in its own attacking direction: centroid depth, spread, length, width, back-line depth, player count and advance per second, plus home-minus-away differences. This stays available while the ball is missing.
- *Ball control (12):* a frame counts as controlled when the ball is visible, within 1.5 m of the nearest player, and moving within 2.5 m/s of that player's velocity. From that we build rolling control shares per team over 0.5 s, 2 s and 6 s each side, and time since and until each team's last and next control. This follows the competition's definition of possession as control, not proximity.

**Step 2a: gradient-boosted evidence (v8).** A histogram gradient-boosting classifier (scikit-learn; 100 iterations, 15 leaves, fixed seed) estimates P(H, A, D | features) from the 72 features. This model is frozen and not retrained for v12.

**Step 2b: temporal network evidence (v12).** The 90 columns are averaged into 5 fps blocks. A dilated temporal convolutional network reads them: a 1x1 convolution to 64 channels, then 7 residual blocks with dilations 1, 2, 4, 8, 16, 32 and 64. The receptive field is 127 blocks each side, which is 25.4 s each side. Its H, A and D probabilities are expanded back to 25 fps and averaged 50/50 with Step 2a's. Section 3 lists the training data and recipe.

**Step 3: decoding (v8).** A three-state Viterbi decoder runs per half on -log P / 25 with a uniform switch penalty of 0.5 and returns the single most coherent H/A/D sequence. Frames outside the two halves are X.

**Why this design.** The ranking metric (spellF1) rewards whole possessions, so the decoder suppresses flicker. An oracle test on other matches showed that giving the model the true team during live play lifted spellF1 from 0.32 to 0.70, far more than perfect dead-ball timing did (0.45). That is why we added team shape, ball control, more ball positions and longer context. The network alone is not enough: on PFF it loses to v8 (-0.031), and the blend carries the gain.

**Compliance with the 4 November clarification.** Every model was trained, and every parameter chosen, on external matches only; the provided data was used only as input or for automatic checks (Section 4.3).

## 2. Assets used and dropped

| Asset | What happened |
|---|---|
| Tactical tracking | **Used.** Main input to every step. |
| Broadcast tracking | **Used (EXPERIMENT-13, v8).** Only to fill missing ball frames (Step 0). Its players and ball are never fed to the model on their own. |
| Smart Tagging | **Tried as model input, dropped (EXPERIMENT-09):** fusing the tagged team into the decoder never helped, even with simulated tags that were 95% correct. **Kept as an automatic check:** agreement between our team calls and the tagged possession team. It was 0.777 (v6), 0.783 (v8) and 0.789 (v12). A new version had to stay above 0.763 (v8's 0.783 minus 0.02, fixed in advance); the check could veto a version but never ranked them. |
| Tactical video | **Tried, dropped (EXPERIMENT-17).** A pretrained ball detector (WASB) filled only 1.7% of graded frames, and its balls were a median 2.24 m from the tracked ball where both existed (limit 1.0 m). The registered stop rule was met, so no video-derived value is in the submission. We trained nothing. |
| Public five-minute labels | Used once for an early version, to catch a home/away inversion, and once each for v8 and v12 as a bug check after the model was fixed (Section 4.3). Never for training, tuning or choosing a version. |

We also tried and dropped, on external matches: temporal stacking (EXPERIMENT-11), larger and seed-averaged models (12), interpolation over short ball gaps (15), a blackout prior (23, -0.0001 on PFF) and a first-touch snap (23, -0.0006 on PFF). Each had a gate fixed before the run. A decoder study (EXPERIMENT-21) produced a candidate that passed its own gate but failed the real-match check in Section 4.3, so it was not offered.

## 3. Training data (none from the provided match)

| Dataset | Use | Labels | Amount |
|---|---|---|---|
| IDSSE/DFL open data (Bassek et al. 2025), CC BY 4.0 | training and validation | The provider's per-frame BallStatus and BallPossession flags. We labelled nothing. | 7 Bundesliga matches, 25 fps |
| SkillCorner open data, MIT | training and validation on real broadcast-tracking noise | SkillCorner's own per-frame possession; no possession is D. | 20 A-League matches, 10 fps upsampled to 25 fps |
| Metrica Sports sample data | training (v12 network only) and diagnostics | Our own fixed event rule (below). | 3 sample games |
| PFF FC World Cup 2022 (access on request from PFF; not redistributed) | testing only | Our own fixed rule applied to PFF's event data. | 64 matches |

The v12 network was trained on SkillCorner 20, DFL 7 and Metrica games 1 to 3. PFF data was never used for training, tuning or choosing between v12 variants; it was the pre-registered adoption test.

**Metrica labels.** Our rule: D from a ball-out, foul or offside stoppage until the restart. Otherwise the label is the team of the last controlled action (pass, carry, recovery, shot or set piece). Duels and touches without control do not change possession. Checks before training: team agreement with the nearest-team call is 0.79 to 0.82 over the 3 games, the D share is 0.32 to 0.41, and the tracking ball is missing on 93% to 99% of D frames and on 0.3% to 1.2% of live frames.

**Training recipe for the network.** Every training match also enters mirrored (pitch flipped, teams swapped, H and A exchanged). Possessions shorter than 3 s get 2 times the weight and those of 3 to 6 s get 1.5 times. SkillCorner frames get weight 0.25. DFL and Metrica matches were also used as degraded copies, so the model sees input as bad as the competition's: ball removed in long runs (41% of frames; median gap 10.7 s, p90 34 s, maximum 221 s), plus position jitter, false ball positions and player dropout. These settings were calibrated only to aggregate ball-visibility statistics of the provided tracking (the 41% figure is also published by the organisers). SkillCorner frames were used only where the ball is visible, because in broadcast video a missing ball mostly signals a stoppage. We trained 12 models (the registered recipe and one ablation, each with 5 folds and a final) on 12 November with fixed seeds and deterministic settings.

The fusion step has no learned parameters. Its constants (1.2 m acceptance, shot length limits, search window) were fixed on simulated DFL second views before any result was seen.

## 4. Validation

Every experiment was registered (data, split, candidates, gates) before its outcome and scored with the official `possession_metrics.evaluate` on full matches. These are not estimates of the competition score.

### 4.1 Model history

**v6 and v8.** In match-grouped 5-fold cross-validation over 27 matches, the v6 model (inside v8) scored 0.460, 0.426 and 0.544 spellF1 under provider labels on the real, gap and clean settings, against 0.249, 0.177 and 0.308 for the organisers' baseline. Provider labels count deflections the competition does not, so later tables use the mean of four conventions (provider labels and controlled-possession rules at 1, 2 and 3 s). Ball fusion (v8) gained +0.043 on simulated second views (7 of 7); on the real feed missing ball fell from 40.7% to 37.6% of graded frames.

**v10, tested and withdrawn.** v10 used a temporal network with about 4.4 s of context each side. Out of fold it beat v8 by +0.026 on the 20 real matches (20 of 20). On the 64 PFF matches v8 scored 0.573 and v10 0.559; v10 won 19 of 64 (match bootstrap 95% interval of the difference -0.020 to -0.008). Its tag agreement on the competition match was also lower (0.772 against 0.783). We withdrew it (`outputs/experiment-19-results.md`).

### 4.2 v12 against v8

EXPERIMENT-22 (`outputs/experiment-22-results.md`) was registered on 12 November, before any v12 code or score. Its gate had five checks, and v12 passed all five. The 64 PFF matches were opened once for it, after the out-of-fold results and both final models existed. Nothing was changed after any held-out score.

The PFF matches were never used for training, tuning or choosing between v12 variants; they were the pre-registered adoption test. But the same 64 matches had been used earlier to compare v8 and v10 (EXPERIMENT-19) and other variants (EXPERIMENTS 20, 21 and 23). So we call them "64 PFF World Cup matches untouched by training", not a fresh holdout.

| spellF1, mean of 4 conventions | Matches | v8 | v12 | v12 minus v8 | Wins |
|---|---:|---:|---:|---:|---:|
| SkillCorner, real tracking, out of fold | 20 | 0.5124 | 0.5277 | +0.0153 | 18 of 20 |
| DFL, simulated gaps, out of fold | 7 | 0.4938 | 0.6120 | +0.1182 | 7 of 7 |
| DFL, clean, out of fold | 7 | 0.6274 | 0.7297 | +0.1023 | 7 of 7 |
| PFF, 64 matches untouched by training | 64 | 0.5729 | 0.6025 | +0.0296 | 62 of 64 |

On PFF the match bootstrap 95% interval of the difference is +0.025 to +0.035, and the four conventions give +0.0327 (raw), +0.0332, +0.0291 and +0.0236 (controlled at 3 s). On SkillCorner they give +0.0159, +0.0158, +0.0156 and +0.0140 (bootstrap of the mean +0.009 to +0.022). On DFL gaps they give +0.1258, +0.1309, +0.1239 and +0.0923. Every convention is positive in every row.

**Ablations.** The network alone scores -0.031 against v8 on PFF (10 of 64 wins), so the blend carries the gain. The same recipe without the Metrica games scores +0.028 on PFF (61 of 64) and +0.019 on SkillCorner, so the PFF gain does not come from the Metrica matches.

**What changed in the predictions (PFF, per match).** v8 predicts 239.5 possessions, v12 252.1, against 278.1 true. Recall of true possessions under 3 s rises from 0.160 to 0.222, and at 3 to 6 s from 0.514 to 0.595. Predicted possession over true dead time is 31.5 for v12 against 29.3 for v8 (v10 had 39.7). The predicted D share is 0.420 against 0.423 (true 0.411). Against the withdrawn v10, v12 is +0.0435 (61 of 64).

Metrica out-of-fold gains are diagnostic only (Game 1 was used for v1, Games 2 and 3 were earlier test sets) and never entered a gate.

### 4.3 Real-match sanity vetoes

We fixed these checks before any v12 score. They use only the provided tracking and the Smart Tagging file, no labels, and none of them tuned anything. They can only veto a candidate: the quiet-stretch check vetoed one (EXPERIMENT-21), and the tag check counted against v10. They are automatic checks of the kind the organisers' 4 November clarification allows.

| Check | Rule | v12 | v8 |
|---|---|---:|---:|
| Two runs | byte-identical CSVs | identical | identical |
| Official validator | passes | passed | passed |
| Possession starts inside quiet stretches (ball missing 5 s or more, mean player speed under 1.5 m/s) | fewer than 27 | 21 | 18 |
| Possessions per minute | not 10% or more above v8 with half the extras in quiet stretches | 2.143 | 2.389 |
| Smart Tagging phase agreement | at least 0.763 | 0.789 | 0.783 |
| Predicted D share | inside the range of v8 and external sources | 0.4752 | 0.4751 |
| Median predicted possession | inside the range of v8 and external sources | 8.86 s | 8.36 s |

Inside the v12 pipeline, v8's decoder on v8's recomputed probabilities reproduces the v8 submission on all 199,319 frames (0 differ). The candidate from the decoder study (EXPERIMENT-21) had 30 starts inside quiet stretches and failed. The withdrawn v10 would have failed the same check with 36.

**Public five-minute window (bug check only).** After v12 was fixed, a pre-registered bug check scored it once on the public five-minute labels: spellF1 0.5677 and macroF1 0.7623, with 14 predicted against 17 true possessions (mean 18.6 s against 14.2 s). v8's single check on 12 November gave 0.5500 and 0.7147. Both were run once, after the model was fixed, and never used to select or tune anything. One possession moves this five-minute score by about 0.06, so the difference between the two is within one possession.

## 5. Possession rules as implemented

- **Who has the ball:** learned from ball proximity, explicit control (ball moving with a player), temporal context and team shape, with up to 25 s of context in the network. A switch needs sustained evidence (penalty 0.5), so touches, deflections and short aerial contests usually do not flip the label. This approximates "controlled possession".
- **Loose balls and missing ball:** the decoder keeps the current team unless the evidence pays for a switch. A missing ball alone is never read as dead.
- **Dead ball:** learned from the providers' ball-status labels and, for the network, our Metrica stoppage rule.
- **Restarts:** handled implicitly as D-to-team transitions. No rule is tied to specific frames or events of the provided match.
- **Abstention:** X only outside the two graded halves.

## 6. Limitations

- **Opposite direction on the competition match.** There v12 predicts 227 possession starts against 253 for v8. On PFF it predicts more than v8 (252 against 240 per match). Apart from the five-minute bug check (Section 4.3), we have no score on the match.
- **The competition truth comes from corrected tracking.** None of our sources is that. SkillCorner and PFF use broadcast tracking, DFL gaps are simulated, and our PFF and Metrica truth is our own rule over event data.
- **Reused data.** The SkillCorner and DFL numbers are cross-validation on data used many times. The PFF matches are untouched by training but were used in earlier comparisons. PFF is one tournament and one provider.
- **DFL gains do not show transfer.** v10 gained +0.09 to +0.11 on DFL and then lost on PFF; v12's DFL gains are the same size, so we rely on PFF and SkillCorner.
- **Reproducibility.** Regeneration needs only the frozen weights and the match data, and was run twice with identical output. PFF was used only as a test set and is not redistributed; retraining needs only DFL, SkillCorner and the Metrica sample games. Training used deterministic settings, but we did not retrain to check that the weights are bit-identical: not checked.
- **Short possessions are still missed.** On PFF v12 finds 22.2% of true possessions under 3 s and 59.5% of those of 3 to 6 s. On earlier benchmarks (v8), resolving them was worth up to +0.29 to +0.36 spellF1 under provider labels. This is the largest known lever.
- **Label conventions differ.** DFL and SkillCorner give mean possessions of 7 to 12 s and 45% to 47% dead time. The organisers' statistics for the five-minute window (mean possession 14.2 s, 19.6% dead) differ, and we did not tune to them. We make no claim about our rank.

## 7. Reproduction

You need Python 3.13 (we froze with 3.13.16) and the organisers' bundle placed as `data/tactical/BAR-ATH_tactical_atd.{txt,xml}` and `data/broadcast/BAR-ATH_broadcast_atd.{txt,xml}`. Two environments: `.venv` from `requirements.txt` (numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1) and `.venv-video` from `requirements-v12.txt` (torch 2.15.0, numpy 2.5.4; on Linux add `--index-url https://download.pytorch.org/whl/cpu`).

```sh
python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
python3.13 -m venv .venv-video && .venv-video/bin/python -m pip install -r requirements-v12.txt
bash v12/run.sh data
```

`v12/run.sh` checks the freeze (`work/v12-evaluation/freeze.json`: model hash, bound source hashes, package versions). It then runs the inputs (v8 fusion path and v6 probabilities), the frozen network, the blend, the decoder and the CSV writer twice, compares the two CSV digests and prints the SHA-256. Files: `v12/run.sh`, `inputs.py`, `predict.py`, `freeze.py`, `freezecheck.py` (freeze checks), `env.py` (version checks), `portability.py` and the frozen weights `v12/final.pt`. The network and training code are in `exp22/` (`net.py`, `views.py`, `metrica.py`).

A Linux cross-platform check runs from `.github/workflows/v12-portability.yml`. Linux result (GitHub Actions run 43799963735; Ubuntu x86_64, Python 3.13.16, torch 2.15.0+cpu, numpy 2.5.4) on 17 public views (7 DFL clean, 7 DFL with gaps, 3 SkillCorner; 2,454,886 frames): the labels match the Mac reference on every frame. The rebuilt inputs differ in the last float bits on all 17 views and the blended probabilities by at most 6.6e-7, which changes no label (`work/v12-evaluation/portability-compare.json`).

**Final file SHA-256 (v12):** `d0c865dc5fde468ebbb57c48a9ad3adfc7dc239aac00db9655fa58f1748ec8e1`. Generated twice on macOS arm64 with Python 3.13.16, byte-identical both times; the official validator passed.

**Base and fallback (v8):** `v8.infer` still regenerates the previous submission, `bb0ba6f55369f862a6ea2151882b82691a7331d6d98733bf6fe6225d08c4034e`, using the commands in `README.md`. Earlier fallbacks: v6 (`v6.infer`), v4 (`v4.infer`), v2 (`v2.verified_inference`). v10 (`v10/run.sh`) still runs; it is withdrawn.

## Credits

- IDSSE/DFL open data: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12:195. doi 10.6084/m9.figshare.28196177.v1, CC BY 4.0.
- SkillCorner open data (MIT): https://github.com/SkillCorner/opendata.
- Metrica Sports sample data: https://github.com/metrica-sports/sample-data. Used for training; labels come from our own event rule.
- PFF FC World Cup 2022: used only for testing, not redistributed, access on request from PFF.
- Organiser starter code (MIT): https://github.com/metrica-sports/more-than-a-hack-2027.

See `THIRD_PARTY.md`.
