# Independent review brief (for Ojas or any reviewer)

Purpose: check that the Mastaadmi submission is what it claims to be. Work only from the public repository https://github.com/nilaymastaadmi/mastaadmi-mtah-2027 and public datasets. **Do not use the competition data bundle, the submission CSV, or any Claude/LLM session that has seen them.** Using Claude on the public repo and public datasets is fine.

Report every item as pass, fail or could-not-check, with the exact command and output. Do not fix anything; report it.

## A. Reproducibility (about 1 hour, needs about 4 GB of public downloads)

1. Fresh clone; Python 3.13; `python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`.
2. `OMP_NUM_THREADS=2 .venv/bin/python -m unittest test_possession test_first_assessment v2.test_v2 v3.test_v3 v4.test_v4 v6.test_v6` must pass (43 tests).
3. Public data: 14 DFL XML files from figshare article 28196177 into `external/idsse/` (skip `*_events_*`), and the SkillCorner `opendata` repo into `external/skillcorner/` with the 20 LFS tracking files into `external/skillcorner-lfs/` (see `v3/skillcorner.py` and `work/EXPERIMENT-05.md` for the exact layout).
4. `OMP_NUM_THREADS=2 .venv/bin/python -m v6.stability_check --compare` must print `digests identical ... True` and `perturbation flips all zero: True`. This proves your machine reproduces the frozen pipeline bit for bit on public data.
5. Optional, slow: `python -m v6.freeze` into a scratch copy must reproduce model sha256 `2e669ecd0aca8d3ac186133d24aa06b8265ecbdc4f6485d75a4d845b6d425add` (training is deterministic).

## B. Claims to verify against the recorded evidence

| Claim (technical note) | Where the numbers live | What to check |
|---|---|---|
| v6 beats v4 on 16/20 SkillCorner matches, +0.015 convention-averaged | `work/v6-development/selection.json`, `rows.jsonl` | Recompute the deltas from rows; confirm the gate logic in `v6/develop.py` matches `work/EXPERIMENT-10.md` |
| v4 beats v2 on 20/20 real matches | `work/v4-development/rows.jsonl` | Same |
| Registration preceded outcomes | git history of `work/EXPERIMENT-*.md` vs result files | Registration commits must be earlier than result commits |
| No competition labels used anywhere | grep the code for `gt`, `public`, `ground_truth`, `evaluation/gt` | Only `starter-repo/evaluation/*` (organiser code) may reference ground truth, and nothing in `v2/`–`v7/` may read it |
| Training labels are provider-produced, never hand-made | `v2/idsse.py`, `v3/skillcorner.py` | Confirm labels come only from BallStatus/BallPossession and `possession.group` |

## C. Rule compliance

- RULES.md: no hand labels on the provided game; no human-produced data about it; repository regenerates the exact file; data not committed. Check `.gitignore` and that no `data/` or CSV is tracked.
- The technical note's training-data table must list every dataset the code reads.

## D. Known gaps (confirm they are disclosed, not hidden)

- All evidence after EXPERIMENT-05 is cross-validation on spent public data, not an untouched holdout.
- v6's real-tracking margin was thin (+0.0154 vs a +0.015 threshold).
- Label-free Smart Tagging agreement for v6 (0.777) is slightly below v4 (0.792).
- Provider label conventions differ from the competition's.

## E. Anything else

Readability of `docs/technical-note.md` for an organiser with 10 minutes; anything overclaimed; anything that would stop them regenerating the file.
