"""Launch with paths relative to this bundle; no laptop-specific home directory."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]


def command(args):
    result = [str(ROOT / '.demo-venv/bin/python'), '-m', 'demo.app', '--port', str(args.port)]
    if args.mode == 'live':
        result += ['--projector', 'laptop', '--training-python',
                   str(args.training_python.absolute()), '--run-dir', str(args.run_dir.absolute())]
    if args.receiver_limit is not None:
        result += ['--receiver', 'nebius', '--receiver-limit', str(args.receiver_limit)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['replay', 'live'])
    parser.add_argument('--training-python', type=Path, default=ROOT / '.inference-venv/bin/python')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'felt-state-model/run')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--receiver-limit', type=int, choices=range(1, 101), metavar='1..100')
    args = parser.parse_args()
    raise SystemExit(subprocess.call(command(args), cwd=ROOT))


if __name__ == '__main__':
    main()
