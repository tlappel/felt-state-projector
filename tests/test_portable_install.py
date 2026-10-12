"""Exercise installation boundaries without downloads, GPU work or API calls."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from portable import install as installer


class InstallerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='felt installer ')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / 'verify_bundle.py').write_text('')
        (self.root / 'felt-state-model').mkdir()
        (self.root / 'felt-state-model/runtime-constraints.txt').write_text('torch==2.9.1+cu130\n')
        for target, value in [('platform.system', 'Linux'), ('platform.machine', 'x86_64')]:
            p = patch('portable.install.' + target, return_value=value)
            p.start(); self.addCleanup(p.stop)
        p = patch.object(installer.sys, 'version_info', (3, 12, 14))
        p.start(); self.addCleanup(p.stop)
        p = patch.object(installer, 'run')
        self.run = p.start(); self.addCleanup(p.stop)
        output = redirect_stdout(io.StringIO())
        output.__enter__(); self.addCleanup(output.__exit__, None, None, None)

    def commands(self):
        return [[str(part) for part in call.args[0]] for call in self.run.call_args_list]

    def test_replay_never_installs_gpu_or_fetches_model(self):
        installer.install('replay', root=self.root)
        commands = self.commands()
        self.assertEqual(commands[0][-1], 'verify_bundle.py')
        self.assertFalse(any('fetch_base.py' in ' '.join(c) or '.inference-venv' in ' '.join(c)
                             for c in commands))
        self.assertEqual(commands[-1][-3:], ['-m', 'pip', 'check'])

    def test_live_validates_before_install_and_checks_gpu_after_download(self):
        installer.install('live', root=self.root)
        commands = self.commands()
        self.assertEqual(commands[1][-1], 'portable/check.py')
        self.assertEqual(commands[-2][-1], 'portable/fetch_base.py')
        self.assertEqual(commands[-1][-2:], ['portable/check.py', '--gpu'])
        self.assertTrue(any(str(self.root / '.inference-venv/bin/python') == c[0]
                            and '--index-url' in c for c in commands))

    def test_plan_does_not_mutate_or_execute_install_commands(self):
        installer.install('live', plan=True, root=self.root)
        self.run.assert_not_called()
        self.assertFalse((self.root / '.demo-venv').exists())

    def test_incompatible_python_fails_before_changes(self):
        with patch.object(installer.sys, 'version_info', (3, 14, 0)):
            with self.assertRaisesRegex(ValueError, 'Python 3.12'):
                installer.install('live', root=self.root)
        self.run.assert_not_called()

    def test_unsupported_os_and_missing_weights_stop_before_install(self):
        with patch.object(installer.platform, 'system', return_value='Windows'):
            with self.assertRaisesRegex(ValueError, 'WSL2'):
                installer.install('replay', root=self.root)
        (self.root / 'felt-state-model/runtime-constraints.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'adapter export'):
            installer.install('live', root=self.root)
        self.run.assert_not_called()

    def test_symlinked_environment_cannot_redirect_installs(self):
        (self.root / '.demo-venv').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            installer.install('replay', root=self.root)
        self.run.assert_not_called()

    def test_existing_environment_wrong_interpreter_rejected(self):
        env = self.root / '.demo-venv'
        (env / 'bin').mkdir(parents=True)
        (env / 'bin/python').write_text('')
        with patch.object(installer.subprocess, 'run', return_value=
                          subprocess.CompletedProcess([], 0, '["/other/environment", [3, 12]]')):
            with self.assertRaisesRegex(ValueError, 'different interpreter'):
                installer.install('replay', root=self.root)
        self.run.assert_not_called()

    def test_failed_verification_stops_before_installation(self):
        self.run.side_effect = subprocess.CalledProcessError(1, ['verify_bundle.py'])
        with self.assertRaises(subprocess.CalledProcessError):
            installer.install('live', root=self.root)
        self.assertEqual(self.run.call_count, 1)
