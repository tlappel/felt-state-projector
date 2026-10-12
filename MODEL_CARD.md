# Felt-State Projector — pilot 002 adapter

This is a narrow body-state-to-language experiment, intended to express supplied
hunger, energy and fatigue as a short first-person whisper. The caller supplies
synthetic values and source IDs. The receiving character supplies its embodiment,
context and interpretation; this component does not choose actions for it.

## Model and training

- Base: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` by NVIDIA.
- Pinned revision: `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`.
- Unmerged LoRA adapter: rank 8, alpha 32, dropout 0; 71 Linear targets,
  8,527,104 trainable FP32 adapter parameters over the frozen BF16 base.
- Synthetic authored examples: 92 training examples across 40 states;
  24 development examples across 10 states. Plain/conversational voices.
- Eight epochs completed; epoch seven selected by development loss.
- Seed 1111, AdamW learning rate 2e-5, microbatch 1, accumulation 4.
- Adapter SHA-256:
  `d11626c0b9517dd81eb4ca2062f71a9c092451a7a221729239f107e685070236`.

[Training source and exact dataset](laptop/pilot-002/README.md) are included for
inspection and reproduction. No retraining is needed to use the downloadable build.
The saved adapter/tokenizer and provenance are inside that build, with original
hashes preserved. Base weights are a separate approximately 7.95 GB download.

## Evaluation and limits

Eight held-out combinations, two voices, two arms: 32 responses. The selected
adapter had 14 faithful responses and two small degree mismatches; the original
base had two faithful, one small mismatch, four clear errors and nine unclear.
All 32 passed mechanical checks. Small mismatches are kept separate from failures.
See [results and review method](docs/RESULTS.md).

These states have now been exposed. The results do not establish general
reliability, superiority over hosted general models, physiological calibration,
individual personalities, or performance on other hardware. This is not a medical
model or a measurement of consciousness. Numeric values are synthetic project
inputs. Future Numen integration and richer individual expression are separate work.

Generation uses greedy decoding, 96 new tokens maximum and `use_cache=False`.
Reported cold startup and warmed generation are different measurements. The
reference laptop used Linux/WSL, Python 3.12 and a 12 GB NVIDIA GPU. The installer
supports that runtime family; other hosts and its automated GPU path are untested.

## License and attribution

Model artifacts, including the adapter and saved tokenizer/configuration, are
provided under the NVIDIA Nemotron Open Model License; see [NOTICE](NOTICE) and
[the included agreement](licenses/NVIDIA-Nemotron-Open-Model-License.pdf).
Original project code and authored synthetic dataset are MIT licensed.
