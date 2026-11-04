# EXPERIMENT-09 results (3 November 2026): gate failed, no Smart Tagging fusion

The study fused team evidence from degraded SkillCorner possession-phase tags (33% coverage, close to the competition's possession codes) into out-of-fold v4 decoding.

Primary score (SkillCorner spellF1, mean over 20 matches × 4 conventions) by tag error rate ε and fusion weight λ:

| ε | λ 0 | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.4969 | 0.4964 | 0.4950 | 0.4905 | 0.4787 | 0.4607 |
| 0.10 | 0.4969 | 0.4975 | 0.4947 | 0.4885 | 0.4781 | 0.4455 |
| 0.20 | 0.4969 | 0.4978 | 0.4946 | 0.4871 | 0.4618 | 0.4028 |

The selected λ is 0 (no fusion). The gate failed, because no weight gains ≥ 0.02 and strong weights hurt even with accurate tags.

**Interpretation.** Phase tags mark clean possession phases where the model is already right. Model errors concentrate in chaotic play, transitions and the box, which the tags do not cover. Imposing the phase team also blurs possession boundaries, which spellF1 scores.

**Caveat.** SkillCorner phases are shorter than Smart Tagging phases, so this is an analogue, not the competition. Fusing tags without validation is therefore not adopted.
