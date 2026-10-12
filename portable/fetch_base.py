"""Explicit multi-GB download of the pinned base snapshot; no inference or training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portable.export_portable_run import MODEL, REVISION

if __name__ == '__main__':
    from huggingface_hub import snapshot_download
    print('Downloading pinned base weights, tokenizer and model code. Allow several GB of disk space.', flush=True)
    print(snapshot_download(repo_id=MODEL, revision=REVISION,
                            allow_patterns=['*.json', '*.safetensors', '*.py', '*.model',
                                            '*.txt', '*.jinja', 'README.md', 'LICENSE*']))
