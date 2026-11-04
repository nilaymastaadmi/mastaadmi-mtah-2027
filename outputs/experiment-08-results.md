# EXPERIMENT-08 results (3 November 2026): gate failed, v4 stays

Decoding study on stored out-of-fold probabilities: 84 candidates × 41 views, scored under 4 label conventions.

- **Selected by the rule:** training R, uniform penalty 1.0, w 1. Primary 0.5017 against the reference (out-of-fold v4) at 0.4969, a delta of +0.0048. SkillCorner wins 15/20.
- **Gate:** failed. The primary delta was +0.0048 (≥ +0.02 required), and DFL contested was −0.011 (≥ −0.01 required). DFL natural (−0.019) and macroF1 (−0.0004) were within tolerance.
- **Top candidates sit within 0.004 of each other,** including asymmetric penalties. The decoder space is flat near v4's setting.
- **Controlled-possession training (K2)** did not beat raw training under any convention average.

## Finding worth keeping

Convention matters for interpretation, not for selection. Out-of-fold v4 SkillCorner spellF1 is:

| Convention | spellF1 |
|---|---:|
| raw | 0.444 |
| controlled1 | 0.481 |
| controlled2 | 0.520 |
| controlled3 | 0.543 |

v4 predicts about 207 possessions per match. If the competition's labels follow the written controlled-possession rule, v4's apparent under-segmentation is largely a provider-label artefact.

## Next

Live-play team assignment remains the main loss (oracle analysis). Smart Tagging team fusion, EXPERIMENT-09, targets it directly.
