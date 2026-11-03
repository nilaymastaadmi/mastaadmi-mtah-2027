# EXPERIMENT-06 results: v4 (team shape, real-noise training), 3 November 2026

Registration: `work/EXPERIMENT-06.md`, committed (77d3d0b) before any outcome. A disclosed probe on spent SkillCorner data motivated it. Evidence is in `work/v4-development/{rows.jsonl,selection.json}` and `work/v4-evaluation/{freeze.json,model.pkl,reference-digests.json}`.

## Result

The selection rule chose **variant C**: v2's 8 features, 29 temporal-context features and 23 team-shape features, with decoder w 1.0 and penalty 0.5. All five checks of the **adoption gate passed**.

| Match-grouped 5-fold CV, mean spellF1 (macroF1) | C | B | A | frozen v2 | baseline50 |
|---|---:|---:|---:|---:|---:|
| SkillCorner, 20 real matches | **0.444** (0.850) | 0.429 (0.843) | 0.371 (0.808) | 0.311 (0.768) | 0.249 (0.702) |
| DFL contested, 7 | **0.414** (0.848) | 0.376 | 0.287 | 0.298 (0.676) | 0.177 |
| DFL gaps, 7 | **0.412** | 0.379 | 0.275 | 0.274 | 0.164 |
| DFL natural, 7 | 0.494 (0.905) | 0.467 | 0.492 | 0.507 (0.888) | 0.308 |

C vs v2:

- SkillCorner +0.133, better on 20 of 20 matches (minimum +0.09);
- DFL contested +0.117, 7 of 7;
- DFL natural −0.013 (3 of 7 better), inside the −0.02 tolerance.

The gains break down as follows:

- Real-noise training alone (A) added about +0.06 on SkillCorner.
- Team shape (B) added about +0.06 more.
- Context (C) added about +0.015.

## Freeze and reproducibility

- C was retrained on all 27 matches (4,762,447 frames; SkillCorner visible-ball frames plus DFL in all conditions). Model sha256 7d23780de776a249bf7195bf848f6ba6f42785c44b11f575ed76586420be5d15.
- `v4/stability_check.py`: zero decoded-frame changes on 19 public reference views under 1e-12 to 1e-8 relative perturbation of probabilities and of all summation-order-dependent features. Digests were stored.
- The local Mac reproduction (`scripts/mac_reproduce.sh`) is pending. v4 replaces v2 only after it passes there.

## Limits

- Cross-validation over spent public datasets, not an untouched holdout. A one-shot PFF test would be run if Nilay provides that data.
- Provider-specific label conventions.
- C drops the baseline prior (w = 1), so it relies entirely on the learned model.
- Same-session implementation and review.
