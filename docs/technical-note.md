# Ball possession from automatic tracking: technical note

More Than A Hack 2027 Phase 1 submission. Team: **Mastaadmi**. Possession file: `Mastaadmi.csv`.

## 1. Approach

The submission labels each of the 199,319 tactical-video frames as H, A, D or X with one fixed pipeline. Every step is an algorithm. No frame of the provided match was labelled or corrected, and no rule or constant is tied to specific frames. The only use of the provided tracking beyond inference was aggregate ball-visibility statistics, used to design a robustness simulation.

1. **Per-frame observations from the provided tracking (72 features).** None uses identities, clock time or club information.
   - **Ball-centric (8):** nearest home and away player distance, ball speed, stationary-ball flag, current missing-gap duration, outside-pitch flag, visibility, and mean player speed.
   - **Temporal context (29):** home-minus-away nearest-distance difference, distance to the nearest line, per-team counts within 5 m and per-team speeds, time since and until the ball is visible, and last/next-visible ball context. It also includes rolling means over ±0.5 s, 2 s and 5 s.
   - **Team shape (23).** For each team, in its own attacking frame: centroid depth, spread, length, width, back-line depth, player count and advance per second, plus home-minus-away differences. Attack direction per half is inferred from median team centroids.

     The team with the ball spreads out and pushes up, while the defending team compacts in front of its goal. This signal stays available while the ball is missing, which is 41% of graded frames.
   - **Ball control (12).** A frame counts as controlled when the ball is visible, within 1.5 m of the nearest player, and moving within 2.5 m/s of that player's velocity (5-frame centred differences). From that: rolling control shares per team over ±0.5 s, 2 s and 6 s, time since and until each team's last and next control, and the last and next controlling team. This follows the competition's definition of possession as *control*, not proximity.
2. **Learned state evidence.** A histogram gradient-boosting classifier (scikit-learn; 100 iterations, 15 leaves, fixed seed) estimates P(H, A, D | features). It was trained only on other matches (Section 3).
3. **Temporal decoding.** A three-state Viterbi decoder runs per half on −log P/25 with a uniform switch penalty of 0.5, returning the single most coherent H/A/D sequence. Frames outside the two halves are X.

**Why this design.** spellF1 rewards whole possessions, so the decoder suppresses flicker.

An oracle analysis on other matches located the main error: substituting the true team during live play lifted spellF1 from 0.32 to 0.70, far more than perfect dead-ball timing did (0.45). Team shape targets exactly that error, including while the ball is missing.

## 2. Assets used and dropped

- **Used:** tactical ATD tracking (`BAR-ATH_tactical_atd.txt`) and EPTS metadata (`_atd.xml`) for team assignment of track slots.
- **Not used:** tactical video, homography, Smart Tagging and broadcast assets. Smart Tagging is promising evidence for team and restarts, but it could not be validated on other matches in time.
- **Public five-minute labels:** used only as a pipeline check, never for training, tuning or choosing between versions.

## 3. Training data (none from the provided match)

| Dataset | Use | Labels: who and how | Amount |
|---|---|---|---|
| IDSSE/DFL open data: Bassek, Rein, Weber & Memmert (2025), *Scientific Data* 12:195, doi 10.6084/m9.figshare.28196177.v1, CC BY 4.0 | training and validation | Per-frame BallStatus and BallPossession from the official DFL position data: dead → D, home → H, guest → A. Labels were produced by the data provider, not by us. | 7 matches, 25 fps |
| SkillCorner open data, github.com/SkillCorner/opendata @ 4340d27, MIT | training and validation on real broadcast-tracking noise | Per-frame team possession from SkillCorner's automatic tracking; no possession → D. | 20 matches, 10 fps upsampled to 25 fps |
| Metrica Sports sample data, github.com/metrica-sports/sample-data @ e706dd5 | earlier versions only | event-derived proxy | 3 matches |

DFL matches were also used in deterministic degraded copies:

- ball removed in long runs (41% of frames; median gap 10.7 s, p90 34 s, maximum 221 s);
- position jitter, false ball positions and player dropout.

These settings were calibrated only to aggregate ball-visibility statistics of the provided tracking. SkillCorner frames were used for training only where the ball is visible, because in broadcast video a missing ball mostly signals a stoppage, which would not transfer to the tactical camera.

## 4. Validation

Every experiment was registered (data, split, candidates, selection rule, pass/fail gates) before outcomes. Scores use the official `possession_metrics.evaluate` on full matches. These are results on other matches, not estimates of the competition score; absolute numbers are not comparable to the competition board.

**Submitted method (v4).** Match-grouped 5-fold cross-validation over 27 matches: each match was scored by a model that never saw it.

| Cross-validated spellF1 | 20 real broadcast matches | 7 DFL matches, contested gaps | 7 DFL matches, clean |
|---|---:|---:|---:|
| **Submitted method (v6)** | **0.512**\* | **0.424**\* | **0.541**\* |
| Previous version (v4) | 0.497\* | 0.414 | 0.494 |
| Earlier version (v2) | 0.311 | 0.298 | 0.507 |
| Organisers' baseline ±50 | 0.249 | 0.177 | 0.308 |

\* Averaged over four label conventions (provider labels, and the competition's controlled-possession rule at 1, 2 and 3 s), because the provider labels count deflections that the competition does not. Under provider labels alone v4 scores 0.444 on real tracking.

v4 beat v2 on 20 of 20 real broadcast matches and 7 of 7 degraded DFL matches. v6 (adding ball control) beat v4 on 16 of 20 real matches (+0.015), 5 of 7 contested (+0.010) and 7 of 7 clean (+0.047), with macroF1 unchanged. Real-tracking macroF1 is 0.85.

**Label-free check on the competition match.** Metrica's automatic Smart Tagging phases carry a team tag. Agreement of our labels with the tagged possession team over BUILD UP, PROGRESSION, ATTACKING TRANSITION and LONG BALL frames was 0.69 for v2, 0.79 for v4 and 0.78 for v6. No ground truth was read; this is a noisy, independent proxy.

**Earlier evidence.** The previous version v2 was tested one-shot on 4 untouched DFL matches and then on 19 untouched real broadcast matches. It beat the baseline and the best tuned non-learning method on every match, which established that the learned approach transfers. A ball-only temporal-context variant trained only on simulated noise did not transfer to real tracking (+0.003) and was rejected; trained together with real noise and team shape, the same features help.

**Caveat.** All public tracking datasets with possession labels known to us were used in earlier tests, so v4's evidence is cross-validation rather than an untouched holdout.

## 5. Possession rules as implemented

- **Who has the ball:** learned from ball proximity, explicit control (ball moving with a player), temporal context and team shape. A switch needs sustained evidence (penalty 0.5), so touches, deflections and short aerial contests usually do not flip the label, approximating "controlled possession".
- **Loose balls and missing ball:** the decoder keeps the current team unless evidence (including team shape) pays for a switch, matching "contested balls stay with the last team".
- **Dead ball:** learned from provider ball-status labels. Cues include a stationary ball, a ball near or outside lines, and play-wide low speed. A missing ball alone is not treated as dead.
- **Restarts:** handled implicitly as D-to-team transitions.
- **Abstention:** X only outside the graded halves.

## 6. Limitations

- **Label conventions:** DFL and SkillCorner definitions of possession and dead ball may differ from the provided ground truth's corrected-tracking convention.
- **No untouched final test:** v4's evidence is 27-match cross-validation.
- **Domain:** Bundesliga and A-League training data; synthetic degradation for DFL; SkillCorner upsampled from 10 fps.
- **Unused information:** Smart Tagging, homography and video. Next steps would be Smart Tagging team and restart evidence, and validation on a new independent dataset.

## 7. Reproduction

Python 3.13 with the pinned `requirements.txt` (numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1). The data bundle goes under `data/tactical/` as in the organisers' DATA.md.

```sh
python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 .venv/bin/python -m unittest test_possession test_first_assessment v2.test_v2 v3.test_v3 v4.test_v4 v6.test_v6
OMP_NUM_THREADS=2 .venv/bin/python -m v6.infer --data-dir data --out Mastaadmi.csv
```

`v6.infer` refuses to run unless the adoption gate passed and the model, settings, every bound source file and the package versions match the freeze. It then validates the CSV with the official validator and prints its SHA-256.

`python -m v6.stability_check --compare` shows that perturbations far larger than cross-platform floating-point differences change no decoded frame, and that outputs on public reference views match the stored digests.

Final file SHA-256: `98d5f2b29d1f1707656da82f3f214ac7e01a99f2714c22f2fbe02c051160ed91` (v6; generated twice on macOS arm64, Python 3.13.16, with identical digests; official validator passed; zero perturbation flips on the competition inputs).

Fallback v4 reproduced locally with SHA-256 `fd8d9430baa68e7d84cb42ef17bdd4cb58b99cca771454151156433c84518837`.

Fallback v2 (`v2.verified_inference`) reproduced locally with SHA-256 `c69eeb177b2d13c35b3235d4226aa1e0827e339c517a9bd4629e53b11535d481`.
