# Evaluation and live demonstration

On October 11, 2026, the original pinned 4B base and selected pilot-two adapter
were evaluated on eight reserved synthetic states in two voices (16 responses
per arm). The rubric was agreed before inference. Hannah reviewed shuffled,
arm-hidden responses before opening the arm key; recognizable formatting could
still suggest origin. Grades were not changed after the reveal. These are human/AI
assisted project judgments, not an independent automated semantic benchmark.

| Model | Faithful | Small degree mismatch | Clear error | Unclear |
| --- | ---: | ---: | ---: | ---: |
| Original 4B base | 2 | 1 | 4 | 9 |
| Selected adapter | 14 | 2 | 0 | 0 |

All 32 outputs passed JSON/source/completion mechanics. The adapter's two small
mismatches were plain-voice descriptions using moderate rather than pretty for
hunger .75 or fatigue .75. Their conversational counterparts were faithful.
Small mismatches are not failures and remain separate from clear contradictions.

[Initial arm-hidden review](../evidence/initial-arm-hidden-review.json) and
[joined comparison](../evidence/comparison-review.json) preserve the exact outputs
and judgments. [Rubric](../laptop/evaluation-002/evaluation-rubric-0.2.md) and
[evaluation plan](../laptop/evaluation-002/evaluation_plan.json) are included.
These states are now exposed; paired voices are not independent states. This
comparison does not establish that the trained model beats a hosted general model.

## Live chain

The laptop projector generated:

> I feel very hungry, with a moderate amount of energy and no tiredness.

The corresponding state was hunger .85, energy .70, fatigue 0. The inspected
whisper was sent to Nemotron 30B-A3B through Nebius Token Factory. The fictional
Rowan receiver responded with its own imagery and a choice to take a food break.
That interpretation belongs to the vessel and is not a projector grading target.

The [user-supplied receipt transcript](../evidence/live-smoke-20261012.json)
records a normal stop, 168 prompt tokens, 68 completion tokens and attempt 1.
It is a transcription, not an imported original event log. No new API call was
made while preparing this repository.

The [fresh-environment laptop result](../evidence/fresh-environment-result.md)
records a separate successful manual installation and generation: 11.36 seconds
startup, 7.10 seconds generation. Earlier warmed runs were about 2.46–2.51 seconds
on laptop power. These observations are not a controlled benchmark or predictions
for other hosts. No new holdout scores are claimed from demo clicks.

## Software verification

The original development project passed 103 offline tests. This curated public
repository includes 31 demo, export, packaging and installer tests:

```bash
python3 -m unittest discover -s tests -v
```

The unchanged replay installer completed a fresh Python 3.12 installation and
Gradio app construction. The packaged live recipe was manually exercised on the
reference laptop, but the new automated live installer has not been GPU-tested.
