# EXPERIMENT-04 results: v2 on untouched DFL matches (2 November 2026)

Registration: `work/EXPERIMENT-04.md`, committed (aebf667) before any outcome. The development and holdout stages each ran once, as approved by Nilay. No code, registration or degradation changed between registration and holdout; the bound-source hashes match across dev.json and holdout.json. Raw evidence is in `work/v2-evaluation/{dev,holdout,provenance}.json`, and the frozen model is `work/v2-evaluation/model.pkl` (sha256 a98dc1f1…56fb).

## Verdict

**Both pre-registered gates passed on all four untouched holdout matches.**

The frozen v2 candidate `blend_learned` uses:

- a gradient-boosting model with v1's eight features, trained on DFL tracking labels from the three development matches;
- a blend with the baseline50 prior at w = 0.75;
- a Viterbi decoder with penalty 0.5.

| Holdout spellF1, mean of 4 matches | natural | gaps | **contested (primary)** |
|---|---:|---:|---:|
| **blend_learned (v2)** | **0.498** | **0.265** | **0.284** |
| hold_majority, window 50 (best non-learning on dev) | 0.391 | 0.214 | 0.216 |
| hmm_hand, penalty 1 (v1 hand emissions, tuned) | 0.418 | 0.193 | 0.186 |
| blend_uniform (smoothing-matched, no information) | 0.319 | 0.184 | 0.199 |
| blend_misaligned (learned q, wrong timing) | 0.322 | 0.183 | 0.196 |
| v1_frozen | 0.320 | 0.176 | 0.193 |
| baseline50 | 0.317 | 0.165 | 0.181 |

- **G1 (promote over v1): PASS.**
  - Contested deltas per match: +0.069, +0.094, +0.092, +0.107 (4/4).
  - Gaps mean: +0.089. Natural mean: +0.178.
  - Contested macroF1: 0.671 vs 0.446 (+0.21 to +0.26).
- **G2 (learned-information edge): PASS.** On contested, blend_learned beats each control on 4/4 matches:
  - uniform: +0.065 to +0.100;
  - misaligned: +0.067 to +0.102;
  - hold_majority: +0.054 to +0.081.

  Unlike on Metrica Game 3, the misaligned and uniform controls no longer explain the gain.

Development results agreed with the holdout: contested means were 0.311 for blend_learned, 0.222 for hold_majority and 0.187 for v1_frozen. Instrument checks passed: nearest-team agreement with the label was 0.75 to 0.81 on every match.

## What this does and does not show

- **The fairest comparison is G2.** v2 trained on DFL labels, while v1 trained on Metrica event proxies, so part of the G1 margin is in-domain training. G2 compares against non-learning controls tuned on the same DFL development matches, and the learned evidence still wins clearly.
- **Much of the gain is dead-ball recognition.** With a clean ball, baseline-style methods almost never predict D (natural macroF1 about 0.40). The learned model recognises stoppages from ball stillness, position and speeds. The DFL BallStatus convention may differ from the competition's corrected-tracking "dead" convention. This is the largest transfer risk.
- **The degradation is synthetic.** It is calibrated to aggregate competition missingness (41% missing, long gaps; players-on-pitch p10 is 18 to 19, matching the competition's 18). It is not the competition's automatic tracking: identity switches and fragmented tracks are only approximated by dropout.
- **Absolute DFL scores are not comparable to the competition.** Contested baseline50 here is 0.181; the published competition full-match baseline50 is 0.2433. No qualifying score, rank, top-20 chance or winning probability follows.
- **The sample is small.** It is four independent holdout matches, all in the Bundesliga, including both first-division matches, and conditions are correlated views of them. With 4/4 wins a sign test gives p = 0.0625. The margins (+0.05 to +0.11 per match) exceed the official per-match noise band of about 0.04, but they are not a significance claim.
- **DFL is now spent.** All seven matches are used. SkillCorner open data (10 A-League broadcast-tracking matches with possession) remains untouched for any future test.
- **Review gap.** The same session implemented, ran and summarised this. No independent review has been done yet; a different-family review is still outstanding.

## Reproduction risk to address before submission

The competition requires that the repository regenerate the identical file. The model was trained on Linux x86_64 under Python 3.13.18. Inference on another platform should give identical gradient-boosting probabilities in almost all cases, but last-bit floating-point differences could flip rare Viterbi ties.

Before submission, run `v2/verified_inference.py` twice on the Mac and confirm the digest is identical. Ideally also confirm it on a second machine. If it is not stable, a probability-rounding amendment must be registered and checked on development data only.
