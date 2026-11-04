# Experiment 09: Smart Tagging team fusion, calibrated on a degraded SkillCorner analogue

Registered 3 November 2026, before any EXPERIMENT-09 outcome. v4 stays the submission unless this gate passes and the fused pipeline is reproduced on the Mac.

## Motivation

- The oracle analysis showed that live-play team assignment is the dominant loss.
- On the competition match, Metrica's possession-phase tags (BUILD UP, PROGRESSION, ATTACKING TRANSITION, LONG BALL; Barcelona = H) agree with v4 on 0.79 of tagged frames, and on about 0.86 of the frames v4 calls alive.
- Given v4's estimated team accuracy, the tags' team label is plausibly about 0.9 or more accurate. This estimate assumes independent errors and is therefore uncertain.
- The tags cover roughly 38% of graded frames, including missing-ball stretches.

## Method (fixed)

- **Model:** the frozen v4 model, unchanged. Out-of-fold v4-recipe probabilities from `external/oof/raw` are used for calibration.
- **Fusion:** for every frame inside a possession tag with team T, add λ/25 to the decoder cost of the other team. Dead ball is untouched. Decoding is then exactly as in v4 (w 1, uniform penalty 0.5).
- **λ grid:** {0, 0.25, 0.5, 1, 2, 4}. λ = 0 is the out-of-fold v4 reference.

## Analogue tags on SkillCorner (fixed)

- **Source:** each match's phases_of_play rows of type build_up, create, direct, quick_break or transition, with the possessing team as the tag team. chaotic, finish and set_play phases are excluded.
- **Frames:** converted from 10 to 25 fps.
- **Degradation, per match and per noise level ε ∈ {0.05, 0.10, 0.20}:**
  - flip each phase's team with probability ε;
  - shift each boundary by Normal(0, 1 s), rounded to frames;
  - use seed sha256(match, ε, 20261103).
- **Scoring:** the official evaluator, under the four label conventions of EXPERIMENT-08 (raw, controlled1, controlled2, controlled3).

## Selection and gate

- **Primary score:** mean SkillCorner spellF1 over 20 matches × 4 conventions, at ε = 0.10.
- **λ selection:** the best primary score; ties within 0.005 go to the smaller λ.
- **Gate (selected λ vs λ = 0):**
  - primary delta ≥ +0.02;
  - wins on at least 14 of 20 matches (convention-averaged) at ε = 0.10;
  - mean delta ≥ 0 at ε = 0.20 (robust if tags are worse than estimated);
  - raw-label macroF1 delta ≥ −0.02 at ε = 0.10.

  DFL has no tag analogue and is not used.

## If the gate passes

1. Add a fused inference entry that loads frozen v4 unchanged, parses Smart Tagging with the fixed code list and mapping, applies the selected λ, and validates the output.
2. Run the cross-machine stability checks.
3. Run two identical Mac runs.

**Label-free sanity check.** EXPERIMENT-07 agreement becomes circular for the fused model. Instead, report fused-vs-v4 agreement on the *unused* attacking codes (BALL IN FINAL THIRD, CREATING CHANCES), descriptively.

## Limits

- SkillCorner phases are derived from SkillCorner's own possession, so the analogue may be optimistic despite the degradation.
- Smart Tagging phases are longer and coarser than SkillCorner's.
- Competition tag accuracy is estimated, not measured.
