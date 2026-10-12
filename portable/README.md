# Felt-State Projector — downloadable test build

Choose **recorded replay** to explore the demo without a GPU, or **live generation**
to run the trained projector. The same ZIP supports both. Replay displays saved
outputs; it does not generate new whispers or call Nebius.

## Requirements

| | Recorded replay | Live generation |
| --- | --- | --- |
| OS supported by this installer | Linux x86_64 / Windows WSL2 | Linux x86_64 / Windows WSL2 |
| Python with venv support | CPython 3.12 or 3.13 | CPython 3.12 |
| GPU | None | NVIDIA with BF16 support and a driver compatible with CUDA 13 |
| Additional setup | Internet for Python packages | C/C++ compiler and Python development headers; internet for packages and base model |
| Model storage | No base-model download | About 7.95 GB of base weights, plus environments and cache |

Live generation has worked on a 12 GB RTX PRO 3000 Blackwell laptop inside WSL,
including freshly installed UI and inference environments. That is a tested
reference, not a guarantee for every 12 GB GPU. Other hosts remain untested.
Native Windows, macOS, ARM and CPU-only live inference are outside this build's
supported setup. On Windows, use the Linux terminal inside WSL2 for all commands.
Plug a laptop into power for inference.

## Download, extract, install

Extract the ZIP into a fresh folder and open a terminal in its
`felt-state-portable/` directory. Do not overlay an older demo.
The current candidate is `felt-state-projector-test-build-20261012.zip`.

Confirm `python3.12 --version`. If that command is unavailable, invoke an existing
Python 3.12 interpreter by its full path instead. The installer does not install
Python, drivers or system build tools. Do not substitute Python 3.14.

For GPU-free replay:

```bash
python3.12 portable/install.py replay
.demo-venv/bin/python portable/launch.py replay
```

For live generation:

```bash
python3.12 portable/install.py live
.demo-venv/bin/python portable/launch.py live
```

Run each command separately; continue only after setup succeeds. Open the localhost
URL printed by the launcher. In live mode, the first **Generate whisper** click
loads the model; later clicks reuse it. Stop the server with **Ctrl+C**.
Stop replay before launching live on the same port.

The installer verifies the bundle, creates `.demo-venv` and (for live mode)
`.inference-venv`, installs pinned dependencies, and runs dependency checks. Live
setup also verifies the adapter, downloads or reuses the pinned base model, and
runs the GPU/tokenizer preflight. Preflight does not execute full generation.
The installer stops on the first error; resolve it and rerun the same command.
Existing matching environments are reused. No administrator privileges are used.

To inspect the commands before installing, append `--plan`:

```bash
python3.12 portable/install.py live --plan
```

The base model and custom model code come from
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, revision
`dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`. Follow its upstream license/access terms.
Base weights are downloaded separately; the ZIP includes the selected unmerged
LoRA adapter and saved tokenizer. The runtime uses pinned Linux x86_64 / Python
3.12 native wheels. Exported dependency constraints are not a complete wheel lock
and do not capture operating-system packages.

## Optional character response through Nebius

The receiver is off by default. With a server-side `NEBIUS_API_KEY` supplied in
your environment, explicitly enable at most one receiver attempt per process:

```bash
.demo-venv/bin/python portable/launch.py live --receiver-limit 1
```

Generate a whisper, inspect it, check the review box, then click **Send to character**.
That click can incur an API charge. There is no startup call or automatic retry.
Never put an API key in the ZIP or repository. This developer option does not
provide judge credentials; free judging access is still being arranged.

The projector supplies bodily context. The receiving vessel owns its body,
interpretation and choices under the world's rules. The fictional Rowan demo
has no Numen memories or ability to perform actions in Numen.

## Evidence and verification

`evidence/fresh-environment-result.md` records the successful manual installation
recipe on the existing laptop and the user-supplied generation receipt. Existing
OS, driver and base cache were reused. That result is distinct from testing this
new setup script on a GPU or on another host. `BUILD_STATUS.json` preserves those
limits. The included live-smoke receipt records an earlier successful Nebius exchange.

Verify the extracted files without installing UI or GPU dependencies:

```bash
python3 verify_bundle.py
python3 -m unittest discover -s tests -v
```

The checksums detect incomplete or changed files; use the separately published
ZIP checksum to check the download. They are not a cryptographic publisher signature.

## Maintainers

`portable/LAPTOP_CHECK.md` documents a completed relocation check with old
environments. It is not part of installing this download. No model export,
training rerun or repeat Nebius call is required for the completed laptop milestone.

In the source project, build a new archive without overwriting earlier versions:

```bash
python3 tools/package_portable.py \
  --model-export /path/to/felt-state-model-20261012.zip \
  --output /path/to/new-test-build.zip
```

Omitting `--model-export` produces a preparation kit without weights, usable for
replay only. Original project code/data use the MIT license. Model artifacts retain the
NVIDIA terms in NOTICE and licenses/. Judge access and optional cloud hosting
remain separate decisions.

## Licensing

Original project code and authored data: [MIT](../LICENSE). Model artifacts have
separate [NVIDIA terms and attribution](../NOTICE); the agreement is included in
`licenses/` at the bundle root. The pinned model card identifies the NVIDIA
Nemotron Open Model License.
