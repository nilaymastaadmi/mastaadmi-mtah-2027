# Development-only amendment, 28 October

Initial development attempt lost to baseline in both conditions. Recorded initial dev.json is retained as-is at work/external-evaluation/dev.json. Holdout has NOT run.

The missing-ball emitter imposed D cost 1.5 versus .5 for each team, contradicting neutral missing evidence. Revised emitter sets all three costs to .5. Reconsider temporal penalty scale using {.1,.5,1.5,5} because initial {5,15,40} heavily suppressed changes at per-second emission scale. These changes are motivated by Game 1 development outcomes; they are not a pre-outcome hypothesis. No changes to event-derived labels, dataset splits, stress masks or promotion gate.

Revision neutral-missing stores separate reports. Initial outputs cannot be reproduced with revised source; retain the source diff and the original registered emitter D=1.5, penalty grid. No initial result is superseded or hidden. Run the revised development, then decide whether to spend holdout; if development still loses, stop this family and retain baseline. If it wins on spellF1 in both conditions and clean macroF1 is no more than .03 lower, freeze the selected setting and run Game 2 once. This adds a development gate before the original holdout gate. Other-match evidence remains a proxy.
