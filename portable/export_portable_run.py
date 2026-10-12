"""Export the reviewed run. Run with the original training venv; no model is loaded.

This file is standalone so it can be downloaded directly to the laptop.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import zipfile

# Frozen identities from the agreed evaluation, not a new model selection.
MODEL = 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16'
REVISION = 'dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f'
ADAPTER_SHA = 'd11626c0b9517dd81eb4ca2062f71a9c092451a7a221729239f107e685070236'
CANONICAL = {
    'result.json': 'b3e065698e8db7890fe1a3efe3efff3e73272679217b6356b13cf7ce85d10d84',
    'dataset.json': '33f9a2d1fc4cdbfe2bf958130276eae7a7d233db34f5e448128ed16f916a11e8',
}
EXACT = {
    'manifest.json': 'e86133b78e4af94deeb9df222b882b28c122cfc3bd31690a1565403caddaa662',
    'adapter_inventory.json': '3661104a76e4920b45657ab82fa6c8455dc6effcb7f7c8d7517c353c9ad2756a',
    'train_mask_audit.json': '7d598ddbc5ffcd8c213dc946f2ed87118af304ed7a4bec0b5d3c1141b06bf514',
    'development_mask_audit.json': 'de0c734596ef4dd85ad2d7edae9615a97ba700ee077d551bc1773e000cf284e4',
}
TOKENIZER_NAMES = {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'chat_template.jinja', 'vocab.json', 'merges.txt',
                   'tokenizer.model'}
REQUIRED = set(CANONICAL) | set(EXACT) | {'best_adapter/adapter_config.json',
                                        'best_adapter/adapter_model.safetensors',
                                        'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json'}
ALLOWED = REQUIRED | {'tokenizer/' + n for n in TOKENIZER_NAMES}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_members(members):
    require(REQUIRED <= members.keys() <= ALLOWED, 'Missing or unexpected run files')
    for name, expected in EXACT.items():
        require(sha(members[name]) == expected, 'Changed reviewed artifact: ' + name)
    for name, expected in CANONICAL.items():
        value = json.loads(members[name])
        canonical = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        require(sha(canonical) == expected, 'Changed reviewed artifact: ' + name)
    require(sha(members['best_adapter/adapter_model.safetensors']) == ADAPTER_SHA,
            'Adapter does not match the selected epoch-seven weights')
    manifest = json.loads(members['manifest.json'])
    require(manifest['model'] == MODEL and manifest['revision'] == REVISION,
            'Wrong base model or revision')
    config = json.loads(members['best_adapter/adapter_config.json'])
    for field, expected in {'base_model_name_or_path': MODEL, 'revision': REVISION,
                            'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 8,
                            'lora_alpha': 32, 'lora_dropout': 0.0, 'bias': 'none'}.items():
        require(config.get(field) == expected, 'Unexpected adapter config: ' + field)
    # The original evaluator performs full target resolution and tokenizer audits
    # at runtime. Export only establishes artifact identity and completeness.
    for name in ('tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json'):
        require(isinstance(json.loads(members[name]), dict), 'Invalid tokenizer file: ' + name)
    return manifest


def read_run(root):
    root = Path(root).resolve(strict=True)
    members = {}
    for name in sorted(ALLOWED):
        path = root / name
        if not path.exists() and not path.is_symlink():
            continue
        require(not any(p.is_symlink() for p in (path, *path.parents) if p != root),
                'Symlink not allowed in export: ' + name)
        require(path.is_file(), 'Expected regular file: ' + name)
        members[name] = path.read_bytes()
    validate_members(members)
    return members


def environment(manifest):
    require(sys.version_info[:2] == (3, 12) and platform.system() == 'Linux'
            and platform.machine() == 'x86_64',
            'Use the original Linux/WSL x86_64 Python 3.12 training venv')
    for name, expected in manifest['versions'].items():
        require(importlib.metadata.version(name) == expected,
                'Training package mismatch: ' + name + '; use the original training venv')
    packages = {}
    for distribution in importlib.metadata.distributions():
        name, version = distribution.metadata['Name'], distribution.version
        require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name)
                and re.fullmatch(r'[0-9][A-Za-z0-9.+!_-]*', version),
                'Cannot safely capture an installed package pin')
        normalized = re.sub(r'[-_.]+', '-', name).lower()
        require(normalized not in packages or packages[normalized] == version,
                'Conflicting installed distributions: ' + normalized)
        packages[normalized] = version
    return {'python': platform.python_version(), 'system': platform.system(),
            'machine': platform.machine(), 'packages': dict(sorted(packages.items())),
            'note': 'Version inventory only, not a wheel lock or proof of a fresh installation.'}


def export_run(root, output):
    members = read_run(root)
    runtime = environment(json.loads(members['manifest.json']))
    payload = {'run/' + name: raw for name, raw in members.items()}
    payload['runtime.json'] = (json.dumps(runtime, indent=2) + '\n').encode()
    payload['runtime-constraints.txt'] = ''.join(
        f'{name}=={version}\n' for name, version in runtime['packages'].items()).encode()
    payload['export_hashes.json'] = (json.dumps({name: sha(raw) for name, raw in payload.items()},
                                              indent=2) + '\n').encode()
    output = Path(output).resolve()
    require(not output.is_relative_to(Path(root).resolve()), 'Write the export outside the saved run')
    # Exclusive creation prevents an accidental overwrite of an earlier export.
    with output.open('xb') as stream:
        try:
            with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
                for name, raw in sorted(payload.items()):
                    archive.writestr('felt-state-model/' + name, raw)
        except BaseException:
            output.unlink()
            raise
    return {'file': str(output), 'bytes': output.stat().st_size,
            'sha256': sha(output.read_bytes()), 'adapter_sha256': ADAPTER_SHA,
            'files': len(payload), 'model_loaded': False, 'network_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('felt-state-model-20261012.zip'))
    args = parser.parse_args()
    try:
        print(json.dumps(export_run(args.run_dir, args.output), indent=2))
    except (ValueError, OSError, KeyError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(1, f'Export stopped: {exc}\nNo model loaded or API call made.\n')


if __name__ == '__main__':
    main()
