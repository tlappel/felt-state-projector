"""Artifact and optional GPU checks. No generation or Nebius call."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from portable.export_portable_run import read_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'felt-state-model/run')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    read_run(args.run_dir)
    if not args.gpu:
        print('Reviewed artifacts present; adapter identity matches. GPU/runtime not checked.')
        return
    from demo.core import evaluation as e
    data, versions, _ = e.validate_saved_run(args.run_dir)
    import torch
    e.require(torch.cuda.is_available(), 'CUDA unavailable; live mode needs a compatible NVIDIA GPU')
    torch.cuda.set_device(0)
    e.require(torch.cuda.is_bf16_supported(), 'BF16 unavailable')
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
    import causal_conv1d
    e.validate_tokenizer(args.run_dir, data)
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({'status': 'runtime preflight passed; full generation still required',
                      'gpu': torch.cuda.get_device_name(0), 'free_vram_bytes': free,
                      'total_vram_bytes': total, 'versions': versions,
                      'model_loaded': False, 'network_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
