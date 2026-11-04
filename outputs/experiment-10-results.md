# EXPERIMENT-10 results (3 November 2026): ball-control features pass, v6 frozen pending the Mac run

Out-of-fold variant D (v4 features plus 12 control features) against the v4 recipe C. Same folds, decoder (w 1, uniform penalty 0.5) and four label conventions.

| Convention-averaged spellF1 | C (v4 recipe) | D (v6) | Delta | Wins |
|---|---:|---:|---:|---:|
| SkillCorner, 20 real matches | 0.4969 | 0.5124 | +0.0154 | 16 / 20 |
| DFL contested, 7 | | | +0.0104 | 5 / 7 |
| DFL natural, 7 | | | +0.0473 | 7 / 7 |

SkillCorner raw macroF1 delta: +0.0044.

**The gate passed all five checks.** The primary margin is thin (+0.0154 against a +0.015 threshold), while the clean-tracking gain is large and consistent.

## Freeze

- D was retrained on all 27 matches (4,762,447 frames). Model sha256 2e669ecd0aca8d3ac186133d24aa06b8265ecbdc4f6485d75a4d845b6d425add.
- `v6.stability_check`: zero flips on 19 public views under 1e-12 to 1e-8 perturbations; digests stored.

v6 replaces v4 only after two identical Mac runs and an EXPERIMENT-07-style tag agreement of at least v4 − 0.02.
