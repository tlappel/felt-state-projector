"""Build the portable developer demo from an explicit artifact allowlist."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'felt-state-demo-20261012'
FILES = ['demo/__init__.py', 'demo/core.py', 'demo/providers.py', 'demo/worker.py',
         'demo/app.py', 'demo/receipts.py', 'demo/README.md', 'demo/requirements.txt',
         'demo/requirements-lock.txt', 'demo/replay.json', 'tests/test_demo.py']
FILES += ['laptop/evaluation-002/' + name for name in (
    'evaluate_pilot_002.py', 'lora_smoke.py', 'evaluation_plan.json',
    'evaluation-rubric-0.2.md')]
VERIFY = '''"""Verify the portable demo without importing UI or ML packages."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent
hashes = json.loads((root / 'bundle_hashes.json').read_text())
for name, expected in hashes.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('Changed or incomplete bundle: ' + name)
from demo.core import ReplayProjector, evaluation
assert len(ReplayProjector().records) == 16
evaluation.validate_package()
print('Demo bundle intact; 16 recorded responses. No model loaded or API call made.')
'''


def main():
    members = {name: (ROOT / name).read_bytes() for name in FILES}
    members['verify_bundle.py'] = VERIFY.encode()
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()}
    members['bundle_hashes.json'] = (json.dumps(hashes, indent=2) + '\n').encode()
    target = ROOT / 'laptop' / (PREFIX + '-r2.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, raw in members.items():
            info = zipfile.ZipInfo(PREFIX + '/' + name, date_time=(2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, raw)
    print(target)
    print('SHA256', hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
