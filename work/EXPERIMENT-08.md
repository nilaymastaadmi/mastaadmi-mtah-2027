# Experiment 08: convention-robust decoding and controlled-possession training

Registered 3 November 2026, before any EXPERIMENT-08 outcome. The frozen v4 stays the submission (`Mastaadmi.csv`, fd8d9430…) unless this experiment's gate passes and the new candidate is reproduced on the Mac.

## Motivation (audit, 3 November)

1. **Label convention.** Provider labels count every deflection as a possession change. The competition (CHALLENGE.md) requires controlled possession, and touches and deflections do not count.

   `v5/labels.py` implements that rule: `controlledK` absorbs an alive run shorter than K seconds that sits between two runs of the other team. On DFL it changes possessions per match from 437 (raw) to 387 (K=1), 324 (K=2) and 283 (K=3); on SkillCorner from 323 to 295, 263 and 245.

   The competition's exact threshold is unknown, so selection must be robust across conventions. Public-window labels are not used.
2. **Decoder search.** v4's decoder was never searched with live/dead switches costing more than team switches; at least one public entry uses that structure.

## Data and folds

Identical to EXPERIMENT-06: 20 SkillCorner and 7 DFL matches, the same 5 folds, and the same training-frame rule. Out-of-fold probabilities come from `v5/oof.py` for two training-label conventions:

- **R:** raw labels, the v4 recipe retrained per fold.
- **K2:** labels relabelled to controlled2 before training.

## Candidates (fixed)

For each of R and K2, a blend of the learned probabilities with the baseline50 prior, w ∈ {0.75, 1}, then one of two decoders:

- **U (uniform):** penalty ∈ {0.25, 0.5, 1, 2, 4}.
- **S (asymmetric):** team penalty ∈ {0.25, 0.5, 1, 2} × live/dead penalty ∈ {0.5, 1, 2, 4}.

Penalties are in cost units of −log p / 25 per frame. The reference is frozen v4 (decoded exactly as frozen).

## Scoring

Every candidate is scored with the official evaluator against the same reference labels under four conventions: raw, controlled1, controlled2 and controlled3.

The **primary score** is the mean SkillCorner spellF1 averaged over the four conventions (20 matches). **Secondary** scores are the DFL contested and DFL natural means over conventions.

## Selection

The candidate with the best primary score wins. Ties within 0.005 go to R before K2, U before S, smaller w and smaller penalties.

## Adoption gate (selected vs frozen v4, out of fold)

All of the following must hold:

- primary mean delta ≥ +0.02;
- SkillCorner wins ≥ 14 of 20 on the convention-averaged per-match score;
- DFL contested convention-averaged mean delta ≥ −0.01;
- DFL natural ≥ −0.02;
- SkillCorner macroF1 (raw labels) delta ≥ −0.02.

If the gate passes:

1. Retrain on all 27 matches with the selected training convention.
2. Freeze with hashes.
3. Run the stability checks and the Mac reproduction.
4. Run the label-free EXPERIMENT-07 tag-agreement check, which must not fall more than 0.02 below v4.

Only then does the candidate replace v4. Otherwise v4 stays.

## Limits

- Cross-validation on spent data.
- The conventions are our implementation of the written rule, not the organisers' labels.
- Smart Tagging restart priors are left for a later registered experiment.

## Amendment (3 November, before any outcome)

The reference changes from frozen v4 to **out-of-fold v4: training R, w 1, uniform penalty 0.5**. Frozen v4 was trained on all 27 matches, so scoring it on them would be in-sample and would bias the gate. The out-of-fold reference is the same recipe and decoder, but never trained on the scored match. Nothing else changes.
