# Pilot 002 training source

The four neighboring files preserve the exact training script, helper files and
92/24-example training/development dataset used for the selected adapter. Their
hash checks intentionally reject edits to the recorded inputs and helpers.

This is provenance and an optional reproduction path; no training is required
for the demo. The portable build includes the selected saved adapter.

With the separately configured Python 3.12 GPU runtime described in START_HERE,
run the script with `--help` and `--check-data` before intentionally training:

```bash
/path/to/.inference-venv/bin/python laptop/pilot-002/pilot_train_002.py --help
/path/to/.inference-venv/bin/python laptop/pilot-002/pilot_train_002.py --check-data
```

See MODEL_CARD.md at the repository root for settings and selection. A new run
will create outputs under `runs/` in the working directory; it will not reproduce
weights bit-for-bit on arbitrary hardware. Existing model artifacts are preserved.
