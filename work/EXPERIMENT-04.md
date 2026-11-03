# Experiment 04 (v2): tuned temporal controls vs learned evidence on untouched DFL matches

Registered 2 November 2026 before any v2 outcome. Status: REGISTERED, NOT RUN. Running the dev and holdout stages requires Nilay's approval. v1 stays frozen and is the fallback. No competition data, labels or public five-minute window are used anywhere in this experiment.

## Question

Experiments 01–03 and the 29 October assessment left three gaps:

1. v1's gain over baseline50 may be smoothing, not learned information.
2. Its proxy truth was event-anchor propagation.
3. Spent Metrica games had no realistic long gaps.

This experiment asks:

- Under competition-like degradation, does any method beat frozen v1 on genuinely untouched matches?
- Does learned evidence beat properly tuned, smoothing-matched non-learning controls?

## Data and provenance

IDSSE/DFL open dataset. Citation: Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12:195. Source: figshare article 28196177 v1 (doi 10.6084/m9.figshare.28196177.v1), licensed CC BY 4.0. It contains seven 2022/23 Bundesliga and 2. Bundesliga matches with official 25 fps optical position data.

Retrieved 2 November 2026 from the figshare API into ignored `external/idsse/`. Fourteen files were fetched: match information and raw observed positions for all seven matches. Every file matched the figshare MD5. Event XML was not fetched or used. Raw data is never committed or redistributed, and attribution is required.

None of these matches has been used by this project before, and none is the competition fixture or involves its clubs.

Before registration, only schema was inspected, on development match J03WOH. It contains:

- a BALL FrameSet carrying BallPossession (1/2) and BallStatus (0/1);
- 105×68 m pitches in centred metres;
- home/guest roles.

A loader smoke test on J03WOH ran without scoring any method. It recorded 137,215 frames, 32 slots, 0 unlabelled in-half frames, and 41.4–41.7% missing ball after degradation.

## Labels

Per frame, BallStatus 0 is D. Otherwise BallPossession 1 is H (home) and 2 is A (guest). Any other code is X (ungraded). This is the same mechanical derivation CHALLENGE.md describes for the competition truth, which comes from corrected tracking. It is a better analogue than event anchors.

DFL's possession definition may differ from the competition's "controlled possession" convention, so these labels remain a proxy. Labels never enter features.

Instrument check, run at the start of each stage before any method is scored: in natural data, nearest-team agreement with the label must exceed 0.5 on every match in that stage. If it does not, the stage aborts and requires an amendment.

## Split, fixed before download

`numpy.random.default_rng(20261102).permutation(sorted(match_ids))` produced the split below. The plain permutation is kept even though both first-division matches fell into the holdout; this is recorded as a domain-shift caveat.

| Role | Matches |
|---|---|
| Development (training and tuning) | J03WOH, J03WOY, J03WPY |
| One-time holdout | J03WMX, J03WN1, J03WQQ, J03WR9 |

The holdout is four independent matches. Conditions are correlated views of those matches, not more matches.

## Conditions

All conditions are deterministic, with seeds from sha256 of match, condition, purpose and 20261102. Training masks and evaluation masks use different seeds.

- **natural**: provider tracking, x/y only.
- **gaps**: ball missing in runs until at least 41% of each half is missing. Run lengths are lognormal with median 10.72 s, p90 33.8 s, clipped to [0.2 s, 221.5 s]. Run starts are uniform among visible frames. Player and ball noise apply as below.
- **contested** (primary): as gaps, but start weight is 1 + 4·[≥3 players within 5 m of the ball] + 4·[ball height ≥ 1.5 m]. Height is used only for masking, never as a feature.
- **Noise in both degraded conditions**:
  - Players: AR(1) jitter (φ 0.96) with σ 0.3 m.
  - Ball: AR(1) jitter with σ 0.5 m.
  - Ball outliers: 1 s runs at uniform random pitch positions, 2% of visible frames.
  - Player dropout: lognormal runs, median 2 s, σ 1, max 60 s, to 8% of on-pitch frames.

Calibration uses only aggregate unlabelled input diagnostics already in `work/first-assessment/domain-audit.json`. No degradation parameter may change after outcomes are seen.

## Methods (all from the same degraded inputs)

- **baseline50**: official starter baseline. Fixed reference.
- **v1_frozen**: hash-verified frozen v1 hybrid (weight 0.25, penalty 0.5), unchanged.
- **majority**: official family; missing ball votes D. Window ∈ {0, 25, 50, 75, 100, 150, 200, 300}.
- **hold_majority**: missing ball abstains (carried forward); an outside-pitch ball votes D. Majority over the same windows.
- **hmm_hand**: frozen v1 hand-designed emission costs with a Viterbi decoder. Penalty ∈ {0.05, 0.1, 0.25, 0.5, 1, 2, 4}.
- **blend_\***: emission −log(w·q + (1−w)·baseline50 prior)/25 with a Viterbi decoder, w ∈ {0.1, 0.25, 0.5, 0.75, 1}, penalty ∈ {0, 0.1, 0.25, 0.5, 1}. Three evidence sources q, each tuned separately on the same grid so every control gets its own best smoothing:
  - **blend_learned**: a new HistGradientBoosting model with v1's hyperparameters and v1's eight features. It is trained on DFL labels across all three training-seed conditions.
  - **blend_uniform**: q = 1/3, the smoothing-matched no-information control.
  - **blend_misaligned**: learned q circularly shifted by half of each half. It keeps the same marginals and autocorrelation but has the wrong timing. This is a control only.

During development, learned q for each development match comes from a model trained on the other two (leave-one-match-out). The holdout model is trained once on all three development matches and hash-frozen.

## Scoring

Scoring uses the official `possession_metrics.evaluate` once per match on the full match: half 1, one ungraded separator frame, then half 2. This follows the competition's aggregation rule, unlike v1's half-average convention. The primary metric is spellF1; macroF1 is secondary.

The v1 secondary PASS-overlap path is not ported.

## Selection (development only, on contested)

For each family, choose the setting with the best mean spellF1 over the three development matches. Ties within 0.005 go to the earliest grid entry.

The candidate is the family with the best development mean among majority, hold_majority, hmm_hand, blend_uniform and blend_learned. Ties within 0.01 go to the earlier family in that simplicity order. blend_misaligned and v1_frozen are never candidates.

The best non-learning control is whichever of majority, hold_majority and hmm_hand has the best development mean. All selections, the frozen model hash and source hashes are written to `work/v2-evaluation/dev.json`.

## Gates (holdout, four matches, run once, no retuning)

**G1, promotion over v1.** All of the following must hold for the candidate versus v1_frozen:

- contested mean spellF1 delta ≥ +0.02;
- the candidate wins on at least 3 of 4 matches on contested;
- gaps mean delta ≥ 0;
- natural mean delta ≥ −0.02;
- contested mean macroF1 delta ≥ −0.03.

If G1 passes, the candidate becomes eligible to be frozen as v2 for local competition inference, which needs separate approval. If G1 fails, v1 remains the candidate.

**G2, learned-information edge.** blend_learned must beat each of blend_uniform, blend_misaligned and the best non-learning control on contested by a mean of at least 0.02 and on at least 3 of 4 matches. If G2 fails, report that no learned edge is established, whatever G1 shows.

Four matches allow no significance claim (a one-sided sign test at 4/4 gives p = 0.0625). The official full-match noise band is about 0.04 per match, so the 0.02 thresholds on means of four are a minimum, not proof.

All rows are reported: every family, condition, match and setting from development, plus every selected family on the holdout, with possession counts.

## Guards

- Each stage writes an exclusive `.started` marker and refuses to run twice.
- The holdout refuses to run if any bound source changed after development: the v2 code, this registration, the v1 modules or the official scorer/baseline.
- The holdout also refuses if the v2 model hash changed.
- v1 pins must still match, because `first_assessment.frozen_model` verifies them before unpickling.

## Limits fixed in advance

- These are synthetic degradations of clean optical tracking, not the competition's automatic tracking.
- The provider possession convention may differ from the competition's.
- Bundesliga is a different domain from the competition match.
- No qualifying score, rank, top-20 chance or winning probability follows from any outcome.
- Tactical tags and video remain unused.
