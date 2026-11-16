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
| No training/fitting/calibrating on ground truth or video | Never. All models trained and all parameters selected on external data via pre-registered cross-validation. The public 5-minute labels were scored only as one-off checks after a model was fixed: v1 (29 Oct, pipeline check), v8 (12 Nov, reality check) and v12 (16 Nov, bug check with a 0.45 line written beforehand, after v12 was already adopted on external evidence). None chose or tuned anything. (Row updated 16 Nov.) |
| Automatic checks on provided data | Smart Tagging agreement (EXPERIMENTS 07, 10) and the broadcast alignment diagnostic (EXPERIMENT-13) are automatic checks. They never tuned anything and never ranked models; they could only veto. From 16 Nov a quiet-stretch guard (thresholds from v8's own output on the provided tracking) vetoed one candidate (EXPERIMENT-21). (Row updated 16 Nov.) |
| Degradation design | EXPERIMENT-04's missing-ball simulation used four aggregate, label-free statistics of the provided tracking: missing share 40.7% (the organisers publish 41%), missing-run median 10.72 s, p90 33.8 s and maximum 221.5 s (listed in `v2/degrade.py`). This shaped external training data; no model parameter was fitted to the provided match. |
| Dataset rights | DFL CC BY 4.0, SkillCorner MIT, Metrica attribution; all credited in THIRD_PARTY.md and the technical note. |

## Consequence

Self-training on the provided match's tracking (algorithm-generated pseudo-labels) was under consideration and is now excluded, because retraining must use only external data. Broadcast-view fusion remains permitted as input use with externally fixed parameters.
