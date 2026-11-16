# EXPERIMENT-21 results: a richer decoder on the probabilities we already have

## Verdict

**PN passed the registered gate but fails the parent's real-match guards, so it is not offered for adoption. v8
stays.** On the competition match PN puts 30 possession starts inside quiet stretches (no ball for 5 s or more,
players nearly still), against 18 for v8 and a limit of under 27 (FINAL CHOICE RULE, `decisions/log.md`,
2026-11-16). That is the same failure pattern that sank v10 (36), at a smaller size. Every other guard passed.

On the external data the gate verdict stands: PN (arm P probabilities + the new decoder) passes, V8N and AN fail.
But the decoder itself adds almost nothing: PN passes because of arm P's probabilities, whose lead over v8 was
already known (EXPERIMENT-19). Against arm P with its old (v10) decoder, the new decoder gains +0.002 on PFF-test
(bootstrap 95% -0.0002 to +0.004, 20 of 32 wins) and loses -0.003 out of fold on SkillCorner real (7 of 20). On v8's
own probabilities the tuned decoder (V8N) gains +0.013 on PFF-64, the same as simply raising v8's penalty to 1.0
(+0.013), and loses -0.0065 on SkillCorner real.

## PN build on the competition match (label-free; guards fixed by the parent before any PN match output)

Model: `work/exp21-evaluation/arm_p_final.pt` (sha256 d80336f9...; EXPERIMENT-19 `final` recipe, 27 external + 64
PFF, trained 12 Nov in 5814 s with 4 threads). Decoder: the registered PN setting (cHA 1.0, cLD 2.0, cDLs 1.0,
cDLo 0.5, bD 0, no minimum durations). Pipeline: `bash exp21/run.sh <data-dir>` (v10.inputs unchanged ->
exp21.predict -> v10.write with the official validator). Freeze: `work/exp21-evaluation/freeze.json`.

| guard (FINAL CHOICE RULE clause 1, plus clause 2 for PN) | PN | v8 | result |
|---|---|---|---|
| two byte-identical full-pipeline runs | abf36a87... twice | | pass |
| official validator | passed (both runs) | | pass |
| starts inside quiet stretches < 27 | **30** | 18 | **FAIL** |
| not (>= 10% more possessions/min with half the extras quiet) | 2.12 vs 2.39 per min (fewer, 29 fewer starts) | | pass |
| Smart Tagging phase agreement >= 0.763 | 0.779 | 0.783 | pass |
| predicted D share within v8 + external truth range (0.283 to 0.542) | 0.493 | 0.475 | pass |
| median possession length within range (3.0 to 10.12 s) | 8.72 s | 8.36 s | pass |
| PN vs v8 under every label convention, SkillCorner OOF and DFL gaps (veto on any loss) | no loss in 8 of 8 cells | | pass |

Overall: **fail** (one guard). PN makes 29 fewer possession starts than v8 but 12 more of its starts sit inside
quiet stretches, so its extra starts are concentrated where the ball is out of sight and play is stopped: the
pattern the guard exists to catch. External truth range for the shape guard: 91 matches (SkillCorner real 20, DFL 7,
PFF 64), `work/exp21-evaluation/external-shape.json`. Quiet-stretch numbers come from `exp19/adoption.py`'s own
functions (thresholds unchanged), written to `work/exp21-evaluation/adoption-match.json`; v8's 18 reproduces the
EXPERIMENT-19 figure.

PN vs v8 per label convention (nested OOF, `work/exp21-evaluation/veto.json`; per-match paired scores for both in
`work/exp21-evaluation/paired-oof.json`):

| setting | raw | controlled1 | controlled2 | controlled3 |
|---|---|---|---|---|
| SkillCorner real (20) | +0.0151 (15 wins) | +0.0140 (15) | +0.0137 (15) | +0.0151 (16) |
| DFL gaps (7) | +0.1271 (7) | +0.1325 (7) | +0.1047 (7) | +0.0655 (7) |
| DFL clean (7, reported) | +0.1069 (7) | +0.1087 (7) | +0.0834 (7) | +0.0386 (6) |

Regeneration needs only the competition data and the frozen weights: `exp21.predict` reads the inputs and
`arm_p_final.pt` (scaler stored inside it); no PFF data. Pinned: Python 3.13.16, torch 2.15.0, numpy 2.5.4
(`requirements-v10.txt`), inference seed 18 and 6 threads with deterministic algorithms (`exp18.arm_a.setup`).
`Mastaadmi.csv` was not touched.


Registration: `work/EXPERIMENT-21.md`, sha256 0b473272 (`work/exp21/registration.sha256`), hashed 2026-11-12 17:28
IST before any new decoder setting was scored. Every number below comes from `work/exp21/gate.json` (built by
`python -m exp21.evaluate gate` from `work/exp21/oof-rows.json` and `work/exp21/pff-rows.json`). The registered rule
was applied unchanged.

## Gate, clause by clause (against v8 as submitted)

| clause | V8N (v8 q, PFF-64) | AN (arm A q, PFF-test 32) | PN (arm P q, PFF-test 32) |
|---|---|---|---|
| G1 PFF mean-of-4 delta >= +0.015 | +0.0130, no | -0.0100, no | +0.1542, yes |
| G2 wins >= 60% | 56 of 64, yes | 13 of 32, no | 32 of 32, yes |
| G3 controlled3 delta >= 0 | +0.0100, yes | -0.0176, no | +0.1376, yes |
| G4 SkillCorner real nested delta >= -0.005 | -0.0065, no | +0.0279, yes | +0.0145, yes |
| G5 DFL gaps nested delta >= -0.005 | +0.0117, yes | +0.1140, yes | +0.1075, yes |
| G6 PFF delta vs same q, own reference decoder >= 0 | +0.0130, yes | +0.0067, yes | +0.0020, yes |
| verdict | FAIL | FAIL | PASS |

Pick: PN is the only passer. PFF-test 32 lower bootstrap bounds against v8 (the registered pick statistic):
V8N +0.0110, AN -0.0184, PN +0.1393.

PFF bootstrap 95% intervals of the delta against v8: V8N +0.0101 to +0.0159 (64 matches); AN -0.0184 to -0.0023;
PN +0.1393 to +0.1682. Harness: every reference decoder reproduced its earlier experiment exactly (max abs spellF1
difference 0.0 on 34 bench views, 64 PFF matches for v8 and v10, 32 for arm P; v8 paths identical to
`v2.methods.viterbi` on 34 of 34 views).

## What the search chose (full search on SkillCorner real + DFL gaps, all 27 views)

| source | cHA | cLD | cDLs | cDLo | bD | Lp (s) | Ld (s) | objective start -> end | settings scored |
|---|---|---|---|---|---|---|---|---|---|
| v8 | 0.25 | 1.5 | 0.75 | 0.5 | 0 | 2.0 | 2.0 | 0.5031 -> 0.5082 | 474 |
| A | 1.0 | 1.5 | 1.0 | 0.5 | 0 | 0 | 0 | 0.5690 -> 0.5750 | 218 |
| P | 1.0 | 2.0 | 1.0 | 0.5 | 0 | 0 | 0 | 0.5637 -> 0.5680 | 370 |

All 18 searches (5 nested folds plus the full one, per source) converged within 4 passes. For v8 every fold chose
Lp = 2.0 s, the largest listed value: the search hit the edge of the registered grid. It was not extended (that would
be a new trial chosen after seeing results). For arms A and P the main change is a dearer entry into dead ball (cLD
0.5 -> 1.5 or 2.0) and a cheaper restart by the other team (cDLo 0.5 vs cDLs 1.0); minimum durations stayed off.

## Decoder contribution alone (new decoder vs the same probabilities with their reference decoder)

| source | SkillCorner real (nested, 20) | DFL gaps (7) | DFL clean (7) | PFF confirmation |
|---|---|---|---|---|
| v8 (ref = v8) | -0.0065 (4 of 20 wins) | +0.0117 (7 of 7) | +0.0071 (5 of 7) | +0.0130 (56 of 64) |
| A (ref = v10 decoder) | +0.0023 (14 of 20) | +0.0077 (7 of 7) | +0.0055 (6 of 7) | +0.0067 (26 of 32) |
| P (ref = v10 decoder) | -0.0028 (7 of 20) | +0.0034 (4 of 7) | +0.0071 (6 of 7) | +0.0020 (20 of 32) |

v8's own P 1.0 control (all four costs 1.0, not a candidate): PFF-64 +0.0131 (54 of 64), PFF-test 32 +0.0145,
SkillCorner real +0.0049, DFL gaps -0.0069, DFL clean -0.0237. So on PFF the tuned decoder ties a one-number change,
and the two trade SkillCorner against DFL in opposite directions.

## Possessions, short possessions, false possessions over dead time (raw convention, per match)

| PFF | predicted | true | recall under 3 s | predicted over true dead time | over the other team | predicted D share (true) |
|---|---|---|---|---|---|---|
| v8 (PFF-64) | 239.5 | 278.1 | 0.160 | 29.3 | 18.9 | 0.423 (0.411) |
| V8N (PFF-64) | 233.8 | 278.1 | 0.167 | 22.1 | 20.8 | 0.423 (0.411) |
| v8 P 1.0 (PFF-64) | 205.1 | 278.1 | 0.112 | 19.2 | n/r | n/r |
| v8 (PFF-test 32) | 241.8 | 276.4 | 0.152 | 32.3 | 19.4 | 0.427 (0.420) |
| v10 (PFF-test 32) | 260.7 | 276.4 | 0.197 | 42.5 | 25.9 | 0.420 |
| AN (PFF-test 32) | 249.8 | 276.4 | 0.185 | 35.2 | 25.7 | 0.420 |
| arm P, v10 decoder | 228.9 | 276.4 | 0.322 | 7.1 | 10.8 | 0.435 |
| PN (PFF-test 32) | 226.0 | 276.4 | 0.317 | 5.8 | 11.2 | 0.434 |

SkillCorner real (nested, 20 matches): v8 215.3 predicted of 334.2 true, recall under 3 s 0.109, 9.2 over dead time;
V8N 221.1, 0.103, 8.1; AN 206.8, 0.114, 4.5; PN 213.9, 0.108, 4.2.

V8N's PFF gain is fewer false possessions over dead time (29.3 -> 22.1 per match) with short-possession recall
about unchanged (0.160 -> 0.167). The new decoder does not recover short possessions on any source; the big
short-possession and dead-time gains on PFF belong to arm P's probabilities.

## Predictions, checked

1. V8N raises cHA above 0.5: **wrong** (0.25; the stiffness came from Lp = 2 s and a dearer entry into D instead).
2. V8N fails G1 with +0.005 to +0.015: **right** (+0.0130).
3. cDLo below cDLs: **right** for all three sources (0.5 against 0.75 or 1.0).
4. Lp at or below 0.4 s: **wrong** for v8 (2.0 s, grid edge), right for A and P (0).
5. Ld above 0: right for v8 (2.0 s), **wrong** for A and P (0).
6. AN fails G1: **right** (-0.0100).
7. PN passes G1 to G5: **right**; G6 passed narrowly (+0.0020).
8. Quiet-stretch rule the most likely failure on the match: **right** (30 against the limit of 27; every other guard passed).

## Limitations

- PFF-64 had been used for decoder looks before (EXPERIMENT-19 diagnostic, EXPERIMENT-20 control), and arm P's lead
  over v8 was spent; PN's pass is not new evidence about arm P. The new evidence is that the decoder adds about
  +0.002 to arm P on PFF and -0.003 on SkillCorner real, both within noise of zero.
- PFF truth is built from PFF game events (EXPERIMENT-19 label rule), not the competition's labels; arm P's PFF gain
  is mostly fewer possessions over dead time, which may be partly PFF-specific.
- The shipped arm P model ('final', 27 external + 64 PFF) is not the model scored on PFF-test (T, 27 + 32 PFF-train);
  the recipe is what was validated, as EXPERIMENT-19 registered for v11. It was trained with 4 threads instead of 5.
- No independent review of the exp21 code: a Fable review was started and stopped at the 12 Nov usage pause, and was not rerun because PN fails a guard and nothing from this experiment ships.
