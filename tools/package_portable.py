"""Build a preparation kit, or a live candidate with a validated laptop export."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from portable.export_portable_run import ALLOWED, REQUIRED, sha, require, validate_members
from tools.package_demo import FILES, VERIFY

EXTRA = ['portable/__init__.py', 'portable/export_portable_run.py', 'portable/check.py',
         'portable/fetch_base.py', 'portable/launch.py', 'portable/requirements-runtime.txt',
         'portable/README.md', 'portable/LAPTOP_CHECK.md', 'tests/test_portable.py', 'tools/package_portable.py',
         'tools/package_demo.py', 'portable/install.py', 'tests/test_portable_install.py',
         'LICENSE', 'NOTICE', 'licenses/README.md', 'licenses/NVIDIA-Nemotron-Open-Model-License.pdf',
         'MODEL_CARD.md', 'docs/RESULTS.md', 'laptop/pilot-002/README.md',
         'laptop/pilot-002/pilot_train_002.py', 'laptop/pilot-002/pilot_examples_002.json',
         'laptop/pilot-002/lora_smoke.py', 'laptop/pilot-002/merge_generation_probe.py']
EXPORT_META = {'runtime.json', 'runtime-constraints.txt', 'export_hashes.json'}


def read_export(path):
    prefix = 'felt-state-model/'
    allowed = {prefix + 'run/' + name for name in ALLOWED} | {prefix + n for n in EXPORT_META}
    required = {prefix + 'run/' + name for name in REQUIRED} | {prefix + n for n in EXPORT_META}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate archive paths')
        require(required <= set(names) <= allowed, 'Missing or unexpected export paths')
        require(sum(i.file_size for i in archive.infolist()) < 512 * 1024 * 1024,
                'Model export unexpectedly large; base weights do not belong here')
        members = {name: archive.read(name) for name in names}
    hashes = json.loads(members[prefix + 'export_hashes.json'])
    require(set(hashes) == {n[len(prefix):] for n in names if n != prefix + 'export_hashes.json'},
            'Export hash inventory mismatch')
    for name, expected in hashes.items():
        require(sha(members[prefix + name]) == expected, 'Export member changed: ' + name)
    validate_members({n[len(prefix + 'run/'):]: raw for n, raw in members.items()
                      if n.startswith(prefix + 'run/')})
    # Constraints are data used by pip later. Admit only package==version lines.
    import re
    constraints = members[prefix + 'runtime-constraints.txt'].decode().splitlines()
    require(constraints and all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][A-Za-z0-9.+!_-]*',
                                             line) for line in constraints), 'Invalid dependency constraints')
    runtime = json.loads(members[prefix + 'runtime.json'])
    require(runtime['system'] == 'Linux' and runtime['machine'] == 'x86_64'
            and runtime['python'].startswith('3.12.'), 'Unexpected captured runtime')
    require(constraints == [f'{name}=={version}' for name, version in sorted(runtime['packages'].items())],
            'Constraints differ from runtime inventory')
    manifest = json.loads(members[prefix + 'run/manifest.json'])
    for name, expected in manifest['versions'].items():
        key = re.sub(r'[-_.]+', '-', name).lower()
        require(runtime['packages'].get(key) == expected, 'Captured package mismatch: ' + name)
    return members


def build(output, model_export=None):
    members = {name: (ROOT / name).read_bytes() for name in FILES + EXTRA}
    members['START_HERE.md'] = (ROOT / 'portable/README.md').read_text().replace(
        '](../LICENSE)', '](LICENSE)').replace('](../NOTICE)', '](NOTICE)').encode()
    for name in ('live-smoke-20261012.json', 'fresh-environment-receipt-20261012.json',
                 'fresh-environment-result.md', 'initial-arm-hidden-review.json', 'comparison-review.json'):
        evidence = ROOT / 'docs/demo-001' / name
        if not evidence.exists():
            evidence = ROOT / 'evidence' / name
        members['evidence/' + name] = evidence.read_bytes()
    if model_export:
        members.update(read_export(model_export))
    status = {'status': 'live_candidate_with_setup_script' if model_export else 'preparation_kit_missing_model_export',
              'model_included': bool(model_export), 'base_weights_included': False,
              'reference_fresh_environment_evidence': 'evidence/fresh-environment-result.md',
              'this_build_live_install_verified': False, 'other_host_verified': False,
              'judge_access_resolved': False,
              'distribution': 'public_repository_test_build'}
    members['BUILD_STATUS.json'] = (json.dumps(status, indent=2) + '\n').encode()
    members['verify_bundle.py'] = (VERIFY + '''
status = json.loads((root / 'BUILD_STATUS.json').read_text())
if status['model_included']:
    from portable.export_portable_run import read_run
    read_run(root / 'felt-state-model/run')
print('Portable status:', status['status'])
print('Reference laptop evidence included; this setup script has not been tested on a GPU.')
print('Other hosts and judge receiver access remain unverified.')
''').encode()
    members['bundle_hashes.json'] = (json.dumps({n: sha(b) for n, b in members.items()}, indent=2) + '\n').encode()
    with Path(output).open('xb') as stream:
        try:
            with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
                for name, raw in sorted(members.items()):
                    info = zipfile.ZipInfo('felt-state-portable/' + name, date_time=(2026, 10, 12, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, raw)
        except BaseException:
            Path(output).unlink()
            raise
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-export', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = args.output or ROOT / 'laptop' / ('felt-state-portable-live-candidate-20261012.zip'
             if args.model_export else 'felt-state-portable-preparation-20261012.zip')
    status = build(target, args.model_export)
    print(json.dumps({'file': target.name, 'sha256': sha(target.read_bytes()), **status}, indent=2))


if __name__ == '__main__':
    main()
