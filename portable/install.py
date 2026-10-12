"""Install the extracted demo into local virtual environments (Linux/WSL)."""
import argparse
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command, root):
    print('+ ' + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=root, check=True)


def check_environment(path, version, root):
    if path.is_symlink():
        raise ValueError(f'{path.name} must be a local directory, not a symlink.')
    if not path.exists():
        return
    python = path / 'bin/python'
    if not python.is_file():
        raise ValueError(f'{path} exists without a usable venv; choose a fresh extraction.')
    probe = subprocess.run([str(python), '-c',
                            'import json,sys; print(json.dumps([sys.prefix, list(sys.version_info[:2])]))'],
                           cwd=root, check=True, capture_output=True, text=True)
    prefix, found_version = json.loads(probe.stdout)
    if Path(prefix).resolve() != path.resolve() or tuple(found_version) != version:
        raise ValueError(f'{path.name} uses a different interpreter; choose a fresh extraction.')


def install(mode, *, plan=False, root=ROOT):
    version = tuple(sys.version_info[:2])
    allowed = [(3, 12)] if mode == 'live' else [(3, 12), (3, 13)]
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('This installer supports Linux x86_64 / Windows WSL2 only.')
    if version not in allowed:
        raise ValueError('Use Python 3.12 for live mode, or 3.12/3.13 for replay. '
                         'Invoke an installed interpreter by its full path if needed.')
    if not (root / 'verify_bundle.py').is_file():
        raise ValueError('Run this installer from the extracted portable ZIP.')
    if mode == 'live' and not (root / 'felt-state-model/runtime-constraints.txt').is_file():
        raise ValueError('Live setup requires the ZIP that includes the adapter export.')

    # absolute(), not resolve(): resolving a venv interpreter loses its environment.
    python = str(Path(sys.executable).absolute())
    ui = root / '.demo-venv/bin/python'
    gpu = root / '.inference-venv/bin/python'
    environments = [root / '.demo-venv']
    if mode == 'live':
        environments.append(root / '.inference-venv')
    commands = [[python, 'verify_bundle.py']]
    if mode == 'live':
        commands.append([python, 'portable/check.py'])
    for env in environments:
        check_environment(env, version, root)
        if not env.exists():
            commands.append([python, '-m', 'venv', str(env)])
    commands += [[ui, '-m', 'pip', 'install', '-r', 'demo/requirements-lock.txt'],
                 [ui, '-m', 'pip', 'check']]
    if mode == 'live':
        constraints = ['-c', 'felt-state-model/runtime-constraints.txt']
        commands += [[gpu, '-m', 'pip', 'install', *constraints, 'torch==2.9.1',
                      '--index-url', 'https://download.pytorch.org/whl/cu130'],
                     [gpu, '-m', 'pip', 'install', *constraints,
                      '-r', 'portable/requirements-runtime.txt'],
                     [gpu, '-m', 'pip', 'check'],
                     [gpu, 'portable/fetch_base.py'],
                     [gpu, 'portable/check.py', '--gpu']]
    if plan:
        print('Plan only: no environments created, packages installed or models downloaded.')
        for command in commands:
            print(shlex.join(map(str, command)))
        return
    for command in commands:
        run(command, root)
    print('\nSetup complete. From the extracted folder, start the demo with:')
    print(f'.demo-venv/bin/python portable/launch.py {mode}')
    if mode == 'live':
        print('The first Generate whisper click loads the model and tests generation.')
    print('The Nebius receiver remains off by default.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['replay', 'live'])
    parser.add_argument('--plan', action='store_true', help='Show commands without installing')
    args = parser.parse_args()
    try:
        install(args.mode, plan=args.plan)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'Setup stopped: {error}', file=sys.stderr)
        print('Resolve the reported error, then rerun the same command. '
              'No automatic retry or system package installation is performed.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
