# EXPERIMENT-05 results: v3 vs frozen v2 on 19 untouched real broadcast-tracking matches (3 November 2026)

Registration: `work/EXPERIMENT-05.md`, committed (a96aff2) before any v3 outcome or holdout download. The loader was designed on pilot match 1925299 only. The 19 holdout matches were downloaded with LFS hash verification and not inspected before the single run. Evidence is in:

- `work/v3-development/{rows.jsonl,selection.json,model.pkl}`;
- `work/v3-evaluation/holdout.json`.

## Verdict

**v2 stays the candidate.** v3 failed its promotion gate. v2 passed its first test on real, non-simulated noisy tracking on all 19 matches.

## Part A: development (DFL, leave-one-match-out, 7 folds)

The registered rule selected variant B: v1's 8 features plus 29 temporal-context features, with v2 model settings, w 0.75 and penalty 0.25. C was 0.0026 higher, inside the 0.01 tie margin, and D chose equal penalties, so it reduced to C.

| Contested condition, mean of 7 folds | spellF1 | macroF1 |
|---|---:|---:|
| B (v3) | 0.335 | 0.774 |
| C | 0.338 | 0.775 |
| A (v2 recipe, retrained per fold) | 0.297 | 0.675 |
| baseline50 | 0.177 | 0.437 |

B beat A on 6 of 7 folds. Under simulated degradation the gain was about +0.04 to +0.06, including +0.06 in the gaps condition.

## Part B: holdout, 19 SkillCorner A-League matches

The data is real broadcast tracking upsampled to 25 fps. Ball missing averaged 40.8%, close to the competition's 40.7%. No match failed the instrument check (minimum agreement 0.70). All methods were frozen, with no tuning on these matches.

| Mean of 19 matches | spellF1 (tracking labels) | spellF1 (phase labels, sensitivity) | macroF1 | predicted / true possessions |
|---|---:|---:|---:|---:|
| v3 | 0.314 | 0.462 | 0.761 | 358 / 334 |
| **v2 (frozen candidate)** | **0.311** | **0.445** | **0.768** | 384 / 334 |
| v1 | 0.266 | 0.381 | 0.701 | 434 / 334 |
| hold_majority ±50 | 0.267 | 0.377 | 0.586 | 452 / 334 |
| baseline50 | 0.248 | 0.351 | 0.701 | 496 / 334 |

- **G1, v3 replaces v2: FAIL.** The mean delta was +0.0026, with 10 of 19 wins (14 required), and macroF1 fell by 0.006. Under the phase-label sensitivity check v3 was +0.018 with 14 wins, but that is not a gate.

  The simulated-noise gain of v3 did not transfer to real noise. The richer temporal features appear to fit the simulator's missingness patterns rather than general structure.
- **T, real-tracking transfer of v2: CONFIRMED.**
  - v2 beat baseline50 on 19/19 matches (mean +0.063; per match +0.025 to +0.077).
  - v2 beat hold_majority on 19/19 (mean +0.044).
  - v2 beat v1 on 19/19 (mean +0.045).

## Interpretation and limits

- This is the strongest evidence so far. It covers 19 independent matches from a different league, a different provider and real tracking noise, and the margin over the baseline is positive on every match. It does not establish the competition score.
- SkillCorner's possession convention differs from the competition's. A missing ball is unusually predictive of stoppages in broadcast tracking because of replays and close-ups. The A-League is not LaLiga. Upsampling from 10 fps smooths positions.
- On these matches, baseline50 happens to average 0.248, near the competition's published full-match baseline50 of 0.2433. That is a coincidence of regimes, not a calibration, and no competition score is inferred from it.
- All public datasets with tracking-derived possession labels that this project knows about are now spent: Metrica 3, DFL 7, SkillCorner 20. Any further model change could only be compared on spent data. A new untouched source would need Nilay's action, for example PFF FC World Cup 2022 data behind a registration form.
- Review gap as before: the same session implemented, ran and summarised this.
