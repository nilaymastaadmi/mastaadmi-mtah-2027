# Experiment 05 (v3): richer temporal features, tested once on real broadcast tracking

Registered 3 November 2026, before any v3 outcome and before any SkillCorner holdout data was downloaded. Nilay asked for continued improvement until the deadline. v1 and v2 stay frozen; v2 is the current candidate and the fallback.

## Part A: development (DFL, diagnostic only)

All seven IDSSE/DFL matches are spent as tests after EXPERIMENT-04. Here they are development data only, with no holdout claim. The protocol is leave-one-match-out over all 7 matches: train on 6, evaluate on 1. Conditions, degradations and seeds are exactly as in EXPERIMENT-04 (natural, gaps, contested), with training-seed masks for training and evaluation-seed masks for evaluation.

The variants are fixed now. No other variant may be added after Part A outcomes without a new registration.

| Variant | Features | Model (HistGradientBoosting) | Decoder |
|---|---|---|---|
| A, v2 recipe | v1's eight features | v2 settings | blend with the baseline50 prior; w ∈ {0.1, 0.25, 0.5, 0.75, 1}, uniform penalty ∈ {0, 0.1, 0.25, 0.5, 1} |
| B | 37 features: the eight plus temporal context (see below) | v2 settings | as A |
| C | as B | 300 iterations, 31 leaves, min leaf 200, learning rate 0.06, l2 1, seed 20261103 | as A |
| D | as C | as C | blend with w ∈ {0.5, 0.75, 1}; team-switch penalty ∈ {0.25, 0.5, 1, 2}; live/dead penalty ∈ {0.1, 0.25, 0.5, 1} |

The extra temporal-context features (`v3/features.py`) are:

- home-minus-away nearest-distance cost;
- ball distance to the nearest pitch line, and distance from the centre lines;
- per-team player counts within 5 m;
- per-team mean player speed;
- time since the ball was last visible and until it is next visible;
- last- and next-visible distance difference and line distance;
- centred rolling means over ±12, ±50 and ±125 frames of distance difference, ball speed, mean player speed, visibility and the stationary flag.

There is no ball height, identity, clock or club information.

Selection:

- Each variant's setting is the one with the best mean contested spellF1 over the 7 folds.
- The v3 candidate is the variant with the best such mean. Ties within 0.01 go to the earlier variant (A < B < C < D).
- If A is selected, v3 is not pursued and v2 stays.

The final v3 model is retrained on all 7 DFL matches with the selected variant and frozen, with its hash recorded, before the holdout.

## Part B: one-time holdout on SkillCorner open data

**Source:** SkillCorner open data, https://github.com/SkillCorner/opendata, commit 4340d274572876239c154c90bc507a9b3250a656, MIT licence © SkillCorner. Attribution is requested; raw data is kept in ignored `external/` only. It contains 20 A-League 2024/25 matches of broadcast tracking at 10 fps. Tracking files are Git LFS objects fetched from GitHub media and verified against their LFS sha256 OIDs.

**Pilot:** match 1925299 was chosen by `numpy.random.default_rng(20261103).choice(sorted(ids))`. It was used to design the loader and label rule, and it is excluded from the holdout. Pilot findings:

- 61,301 frames; detected ball in about 63% of in-play frames;
- players extrapolated to about 22, with about 12 detected at the median;
- the per-frame possession group agrees with SkillCorner phases of play on 87% of in-play frames.

**Holdout:** the 19 other matches. None has been downloaded or inspected yet.

**Loader (fixed):**

- Periods come from `match_periods` in match.json.
- Data is upsampled to 25 fps per period: frame k sits at 10 fps position k·10/25.
- Player and ball x/y are linearly interpolated between the neighbouring 10 fps frames when both are finite, otherwise NaN.
- Players use SkillCorner's extrapolated positions, with slots by player_id and teams from match.json.
- The ball is kept only where `is_detected` is true; otherwise it is NaN.
- Metres are normalised by the match pitch size.
- Each match is assembled as half 1, one ungraded separator frame, then half 2, and scored once with the official `possession_metrics.evaluate`.

**Labels (fixed):** the nearest 10 fps frame's `possession.group`: home team is H, away team is A, none is D.

Sensitivity analysis, reported but not used for gates: labels from phases of play, with frames inside a phase taking the possessing team and all others D.

**Caveats fixed in advance:**

- SkillCorner's possession definition differs from the competition's.
- Broadcast cuts during stoppages make a missing ball unusually predictive of D.
- Detected players are fewer than in tactical-camera tracking.

**Instrument check:** on each holdout match, natural nearest-team agreement with H/A labels must exceed 0.5, or the match is excluded and reported. If more than 3 matches fail, the holdout aborts and needs an amendment.

**Methods (no tuning on SkillCorner):**

- v3 (frozen, Part A);
- v2 frozen (w 0.75, penalty 0.5);
- v1 frozen;
- baseline50;
- hold_majority window 50, the best non-learning family from EXPERIMENT-04 development.

All run on the same real inputs (one condition: real tracking as provided).

## Gates (19 holdout matches, run once)

- **G1, v3 replaces v2:**
  - mean spellF1 delta (v3 − v2) > 0;
  - v3 wins on at least 14 of 19 matches (a one-sided sign test gives p ≈ 0.03);
  - mean macroF1 delta ≥ −0.03.

  If G1 fails, v2 stays the candidate.
- **T, transfer of the current candidate, reported regardless of G1:** whether v2 beats baseline50 and hold_majority on at least 14 of 19 matches. This is the first test of any of our models on real, non-simulated noisy tracking.

All per-match rows are reported, along with the phases-of-play label sensitivity. Nothing about qualification, rank or winning probability follows.

## Guards

- One-shot `.started` markers.
- The holdout binds the hashes of the v3 sources, this registration, the frozen v3 model and the v1/v2 sources.
- The v2 freeze check and v1 pins must pass.
- Same review gap as before: the implementer runs and summarises; independent review is outstanding.
