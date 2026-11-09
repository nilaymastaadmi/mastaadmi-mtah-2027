# Organiser rules clarification (received by email, 4 November 2026) and our compliance

The organisers wrote to all participants that the following takes precedence over earlier documentation and answers.

Not allowed: checking output against the video; hand labels or any manually created data on the provided game; uploading the provided data to AI services; training, fitting or calibrating the algorithm on the ground-truth possession data or on the provided video.

Allowed: automatic checks using only the provided data; inspecting the data and video to understand possession; using the video, tracking, Smart Tagging or other generated data as input, provided the algorithm was not trained on the ground truth or the video; intermediate files; any dataset with usage rights.

Training and tuning must use only external data.

## Compliance record

| Rule | Our practice |
|---|---|
| No checking against video | Never done; the video was never downloaded. |
| No hand or human-made labels | None. Training labels are provider-generated on other matches (DFL BallStatus/BallPossession, SkillCorner possession group). |
| No upload to AI services | None. All competition-data steps run on Nilay's Mac; only aggregate statistics leave it. |
| No training/fitting/calibrating on ground truth or video | Never. All models trained and all parameters selected on external data via pre-registered cross-validation. The public 5-minute labels were never scored by any version after v1's one-time pipeline check. |
| Automatic checks on provided data | Smart Tagging agreement (EXPERIMENTS 07, 10) and the broadcast alignment diagnostic (EXPERIMENT-13) are automatic checks. They never selected a model; they confirmed selections already made on external data. |
| Degradation design | EXPERIMENT-04's missing-ball simulation used aggregate ball-visibility statistics of the provided tracking (the 41% figure is also published by the organisers). This shaped external training data; no model parameter was fitted to the provided match. |
| Dataset rights | DFL CC BY 4.0, SkillCorner MIT, Metrica attribution; all credited in THIRD_PARTY.md and the technical note. |

## Consequence

Self-training on the provided match's tracking (algorithm-generated pseudo-labels) was under consideration and is now excluded, because retraining must use only external data. Broadcast-view fusion remains permitted as input use with externally fixed parameters.
