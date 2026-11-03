# Experiment 06 (v4): team-shape features and real-noise training, cross-validated

Registered 3 November 2026, before any v4 outcome. v2 stays the frozen submission candidate unless this experiment's adoption gate passes.

## Why (development evidence, disclosed)

Two diagnostics on spent SkillCorner matches motivated this experiment. They are development evidence, not tests.

1. **Oracle substitution** on 10 matches, applied to v2. Mean spellF1 was:
   - 0.319 for v2 as it stands;
   - 0.445 with true dead-ball segmentation;
   - 0.376 with true restart teams;
   - **0.702 with the true team during live play.**

   The bottleneck is the team in possession during live play.
2. **Probe:** train on 10 SkillCorner matches (all frames), test on the 9 others (`v4/shape.py`, v2 decoder w 0.75, p 0.5).
   - v2 features: 0.355.
   - v2 features plus team shape: 0.419, better on 9 of 9 matches.

No untouched tracking-possession data remains (Metrica, DFL and SkillCorner are all spent). The evidence standard here is therefore match-grouped cross-validation, stated as such. If Nilay supplies an untouched source (for example PFF FC World Cup 2022), the selected v4 must also pass a one-shot test there before submission.

## Data

- **DFL, 7 matches.** natural, gaps and contested conditions exactly as in EXPERIMENT-04, with training-seed masks for training and evaluation-seed masks for evaluation.
- **SkillCorner, 20 matches (pilot included).** Real tracking via `v3/skillcorner.py`, unchanged.

**Training uses only SkillCorner frames with a visible ball.** In broadcast data a missing ball mostly means a stoppage, which would not transfer to the competition's tactical camera. Evaluation uses all frames.

## Folds (fixed)

Five folds. The SkillCorner match at sorted-id index i goes to fold i mod 5, and the DFL match at sorted-id index j goes to fold j mod 5. Each fold's matches are evaluated by models trained on the other four folds only.

## Variants (fixed)

All variants use HistGradientBoosting with v2 settings and the v2 blend decoder with the baseline50 prior.

| Variant | Features |
|---|---|
| A | v2's 8 features, retrained on this data (isolates the real-noise training effect) |
| B | A plus 23 team-shape features (`v4/shape.py`) |
| C | B plus the 29 temporal-context features of `v3/features.py` |

The team-shape features are, per team: centroid, spread, length, width, back-line depth in the attacking frame, player count and advance per second, plus home-minus-away differences.

The decoder grid has w ∈ {0.5, 0.75, 1} and penalty ∈ {0.1, 0.25, 0.5, 1}. Each variant's setting is chosen by the mean SkillCorner cross-validation spellF1; ties within 0.005 go to the earliest grid entry.

Reference: frozen v2. It never saw SkillCorner, but it was trained on DFL J03WOH, J03WOY and J03WPY.

## Selection

The variant with the best mean SkillCorner CV spellF1 wins. Ties within 0.01 go to the simpler variant (A < B < C).

## Adoption gate (selected variant vs frozen v2)

All of the following must hold:

- **SkillCorner (20 matches):** mean spellF1 delta ≥ +0.03, on at least 15 of 20 matches, and mean macroF1 delta ≥ −0.02.
- **DFL contested (7 matches):** mean spellF1 delta ≥ 0.
- **DFL natural:** mean spellF1 delta ≥ −0.02.

If the gate passes, the selected variant is retrained on all 27 matches with the same rule (SkillCorner visible-ball frames plus DFL in all conditions), frozen with hashes, and given a verified inference entry. It must then pass the cross-machine stability check and the Mac reproduction before it replaces v2 as the submission, plus a one-shot PFF test if that data becomes available.

If the gate fails, v2 stays.

## Limits

- Cross-validation on spent data, not an untouched holdout.
- SkillCorner labels are provider-specific; DFL degradation is simulated.
- Team-shape attack direction is inferred per half from median team centroids.
- No qualifying score or rank follows from any of this.
