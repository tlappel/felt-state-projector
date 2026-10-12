"""Portable handoff tests use synthetic weights; no GPU/network or real export."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from portable import export_portable_run as e
from portable.launch import command
from tools.package_portable import read_export, build


class PortableTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.run = self.root / 'source'
        self.run.mkdir()
        manifest = {'model': e.MODEL, 'revision': e.REVISION, 'versions': {'torch': '2.9.1+cu130'}}
        self.runtime = {'python': '3.12.14', 'system': 'Linux', 'machine': 'x86_64',
                        'packages': {'torch': '2.9.1+cu130'}}
        config = {'base_model_name_or_path': e.MODEL, 'revision': e.REVISION,
                  'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 8,
                  'lora_alpha': 32, 'lora_dropout': 0.0, 'bias': 'none'}
        self.members = {name: b'{}' for name in e.REQUIRED}
        self.members['manifest.json'] = json.dumps(manifest).encode()
        self.members['best_adapter/adapter_config.json'] = json.dumps(config).encode()
        self.members['best_adapter/adapter_model.safetensors'] = b'SYNTHETIC WEIGHTS, NOT A MODEL'
        exact = {name: e.sha(self.members[name]) for name in e.EXACT}
        canonical = {name: e.sha(b'{}') for name in e.CANONICAL}
        for name, value in [('EXACT', exact), ('CANONICAL', canonical),
                            ('ADAPTER_SHA', e.sha(self.members['best_adapter/adapter_model.safetensors']))]:
            p = patch.object(e, name, value); p.start(); self.addCleanup(p.stop)
        for name, raw in self.members.items():
            path = self.run / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)

    def export(self):
        dest = self.root / 'model.zip'
        with patch.object(e, 'environment', return_value=deepcopy(self.runtime)):
            receipt = e.export_run(self.run, dest)
        return dest, receipt

    def test_allowlisted_export_roundtrip_and_no_overwrite(self):
        (self.run / '.env').write_text('NEBIUS_API_KEY=DO_NOT_COPY')
        (self.run / 'tokenizer/.env').write_text('DO_NOT_COPY')
        archive, receipt = self.export()
        members = read_export(archive)
        self.assertEqual(receipt['adapter_sha256'], e.ADAPTER_SHA)
        self.assertFalse(any(b'DO_NOT_COPY' in raw for raw in members.values()))
        self.assertFalse(receipt['model_loaded'])
        with self.assertRaises(FileExistsError): self.export()

    def test_tampered_weight_or_missing_tokenizer_stops_export(self):
        weight = self.run / 'best_adapter/adapter_model.safetensors'
        weight.write_bytes(b'WRONG')
        with self.assertRaisesRegex(ValueError, 'Adapter does not match'): e.read_run(self.run)
        weight.write_bytes(self.members['best_adapter/adapter_model.safetensors'])
        (self.run / 'tokenizer/tokenizer.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'): e.read_run(self.run)

    def test_symlinked_artifact_rejected(self):
        target = self.run / 'tokenizer/tokenizer.json'; target.unlink()
        outside = self.root / 'outside.json'; outside.write_text('{}'); target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Symlink'): e.read_run(self.run)

    def test_import_rejects_extra_paths_and_modified_member(self):
        path, _ = self.export()
        with zipfile.ZipFile(path) as z: original = {n: z.read(n) for n in z.namelist()}
        for label, changes in [('path', {'felt-state-model/../../escape': b'x'}),
                               ('changed', {'felt-state-model/run/tokenizer/tokenizer.json': b'{"changed":true}'})]:
            candidate = self.root / (label + '.zip')
            with zipfile.ZipFile(candidate, 'w') as z:
                for name, raw in (original | changes).items(): z.writestr(name, raw)
            with self.assertRaises(ValueError): read_export(candidate)

    def test_import_rejects_dependency_option_injection_even_if_rehashed(self):
        path, _ = self.export()
        with zipfile.ZipFile(path) as z: members = {n: z.read(n) for n in z.namelist()}
        name = 'runtime-constraints.txt'
        members['felt-state-model/' + name] = b'--extra-index-url https://example.invalid\n'
        hashes = json.loads(members['felt-state-model/export_hashes.json'])
        hashes[name] = e.sha(members['felt-state-model/' + name])
        members['felt-state-model/export_hashes.json'] = json.dumps(hashes).encode()
        bad = self.root / 'bad-pip.zip'
        with zipfile.ZipFile(bad, 'w') as z:
            for name, raw in members.items(): z.writestr(name, raw)
        with self.assertRaisesRegex(ValueError, 'constraints'): read_export(bad)

    def test_launcher_preserves_venv_symlink_and_receiver_off_default(self):
        venv = self.root / 'python'; venv.symlink_to(sys.executable)
        args = argparse.Namespace(mode='live', port=7860, training_python=venv,
                                  run_dir=self.run, receiver_limit=None)
        cmd = command(args)
        self.assertEqual(cmd[cmd.index('--training-python')+1], str(venv))
        self.assertNotIn('--receiver', cmd)
        args.receiver_limit = 1
        self.assertEqual(command(args)[-4:], ['--receiver', 'nebius', '--receiver-limit', '1'])

    def test_package_modes_distinguish_missing_weights_and_unverified_live(self):
        for included in (False, True):
            model, _ = self.export() if included else (None, None)
            output = self.root / ('live.zip' if included else 'kit.zip')
            status = build(output, model)
            self.assertEqual(status['model_included'], included)
            self.assertFalse(status['this_build_live_install_verified'])
            with zipfile.ZipFile(output) as z:
                names = z.namelist()
                self.assertIn('felt-state-portable/START_HERE.md', names)
                has_weights = any(n.endswith('adapter_model.safetensors') for n in names)
                self.assertEqual(has_weights, included)
                hashes = json.loads(z.read('felt-state-portable/bundle_hashes.json'))
                for name, expected in hashes.items():
                    self.assertEqual(e.sha(z.read('felt-state-portable/' + name)), expected)

    def test_runtime_mismatch_fails_before_collecting_environment(self):
        with patch.object(e.sys, 'version_info', (3, 12)), patch.object(e.platform, 'system', return_value='Linux'), \
             patch.object(e.platform, 'machine', return_value='x86_64'), \
             patch.object(e.importlib.metadata, 'version', return_value='CHANGED'):
            with self.assertRaisesRegex(ValueError, 'package mismatch'):
                e.environment({'versions': {'torch': '2.9.1+cu130'}})
