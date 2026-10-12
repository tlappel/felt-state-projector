# Felt-State Projector

**A bodily whisper. A character's own response.**

Felt-State Projector turns supplied hunger, energy and fatigue into a short,
expressive description. A locally fine-tuned NVIDIA Nemotron 4B model produces
the whisper; an optional Nemotron 30B-A3B receiver on Nebius interprets it in a
fictional scene. The projector delivers bodily context. The receiving character
owns its body, interpretation and choices.

This is a standalone Numen-adjacent experiment. The demo has no personal memories,
world integration or ability to execute actions in Numen.

## Try the downloadable test build

[**Download the test-build ZIP**](https://github.com/tlappel/felt-state-projector/raw/refs/heads/main/downloads/felt-state-projector-test-build-20261012.zip)
(approximately 34 MB) · [SHA-256](downloads/felt-state-projector-test-build-20261012.zip.sha256)

Extract it, open a Linux/WSL terminal in `felt-state-portable/`, and follow
`START_HERE.md`. It includes the setup script, saved adapter/tokenizer and recorded
examples. The base model downloads separately for live mode.

| Mode | Requirements | What happens |
| --- | --- | --- |
| Recorded replay | Linux x86_64 / WSL2, Python 3.12 or 3.13; no GPU | Displays saved evaluation outputs |
| Live projector | Linux x86_64 / WSL2, Python 3.12, compatible NVIDIA BF16 GPU | Generates with the saved adapter |
| Optional receiver | Live setup plus your server-side Nebius key | Sends an inspected whisper to the hosted character model |

For replay, from the extracted build:

```bash
python3.12 portable/install.py replay
.demo-venv/bin/python portable/launch.py replay
```

For live generation, replace `replay` with `live` in both commands. Live setup
requires a driver compatible with the pinned CUDA 13 runtime, build tools and
several GB for the base model and dependencies. Native Windows/macOS/ARM and
CPU-only live inference are outside the current supported setup.
See [full installation instructions](portable/README.md).

**Status:** local live generation and a real Nebius receiver exchange have worked;
fresh UI/inference environments worked on the reference 12 GB RTX PRO 3000 laptop.
The replay installer was tested end to end. The automated live installer and other
GPU hosts remain untested. This is a downloadable test build, not a hosted demo.
No API key is included; free judging receiver access is still being arranged.

## What we trained and measured

The selected epoch-seven LoRA adapter uses
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` at an exact pinned revision. Training used
92 synthetic examples, with 24 development examples. In the eight-state/two-voice
final comparison, the adapter produced 14 faithful responses and two small degree
mismatches; the original base produced two faithful, one small mismatch, four
clear errors and nine unclear responses. All 32 passed mechanical checks.

Small mismatches are not failures. This narrow result does not establish broad
reliability, individualized expression or superiority to hosted general models.

- [Model, training and limitations](MODEL_CARD.md)
- [Evaluation and live evidence](docs/RESULTS.md)
- [Training source and exact dataset](laptop/pilot-002/README.md)
- [Nebius request and response transcript](evidence/live-smoke-20261012.json)

## Develop and verify

This repository contains the demo, isolated GPU worker, installer, training source,
and selected evidence. To run its 31 offline tests, no GPU or API key is needed:

```bash
python3 -m unittest discover -s tests -v
```

For runtime use, download and extract the test build rather than installing from
the repository root. The ZIP contains the adapter and integrity manifest expected
by the installer. The packaging scripts use explicit file inventories.

Created by Travis Appel with OpenAI ChatGPT/Codex (Hannah) assistance. Models are
by NVIDIA; hosted receiver inference uses Nebius Token Factory.

## License

Original code, documentation and authored synthetic data: [MIT](LICENSE).
The adapter and saved NVIDIA tokenizer/configuration artifacts are subject to
separate [NVIDIA terms and attribution](NOTICE), included inside the ZIP too.
The base weights are not redistributed here. Dependencies retain their own licenses.
