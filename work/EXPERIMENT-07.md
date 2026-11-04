# Experiment 07: label-free Smart Tagging corroboration of v4 vs v2 on the competition match

Registered 3 November 2026, before any v4-vs-v2 comparison on competition inputs. Local Mac only; aggregate output only. No ground truth or public-window labels are read.

## Rationale

The first audit run (v2 only) established the tag semantics.

- The Team tag is the team in possession for attacking codes: BUILD UP, PROGRESSION, ATTACKING TRANSITION, LONG BALL, BALL IN FINAL THIRD, CREATING CHANCES, BALL IN THE BOX, PLAYERS IN THE BOX.
- It is the defending team for the mirrored DEFENDING IN ... and DEFENSIVE TRANSITION codes.
- The mapping is Barcelona = H (ATD team 0) and Athletic = A.

Smart Tagging is generated automatically by Metrica, independently of our models. Agreement with it is therefore a label-free (but noisy) proxy for team-in-possession accuracy on the actual match.

## Measure (fixed)

**Phase agreement.** Use the four non-redundant possession codes BUILD UP, PROGRESSION, ATTACKING TRANSITION and LONG BALL. They are the codes whose agreement with nearest-player ownership was at least 0.6 in the first audit, and their mirrored defending codes are excluded as duplicates. For each, count the frames inside tag intervals where the model's label equals the tagged possession team, out of all graded frames inside those intervals, pooled across codes. Report v2 and v4.

## Decision rule (fixed)

- v4 phase agreement ≥ v2 phase agreement − 0.02: corroborated. Keep v4.
- v4 lower than v2 by more than 0.02: red flag. Escalate to Nilay with both files and keep v2 as candidate until resolved.

Other per-code numbers are reported descriptively only. This check cannot promote any new model.
