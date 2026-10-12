"""Meaningful demo boundary tests; no actual GPU or paid receiver calls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from demo.core import DemoError, DemoService, ReplayProjector, packet_for, evaluation
from demo.providers import LaptopProjector, NebiusReceiver, NoRedirect
from demo.receipts import recover


class Receiver:
    def __init__(self, fail=False):
        self.requests = []
        self.fail = fail

    def respond(self, request):
        self.requests.append(deepcopy(request))
        if self.fail:
            raise RuntimeError('PRIVATE error payload')
        return {'text': 'Synthetic receiver test output.'}


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.receiver = Receiver()
        self.now = 10
        self.service = DemoService(ReplayProjector(), self.receiver,
                                   log_dir=self.root / 'events', clock=lambda: self.now)

    def generate(self, owner='browser-a'):
        return self.service.generate(owner, 'focus-final-01', 'plain')

    def send(self, c, **kwargs):
        args = {'owner': 'browser-a', 'token': c['token'], 'case_id': 'focus-final-01',
                'voice': 'plain', 'inspected': True}
        args.update(kwargs)
        return self.service.deliver(**args)

    def test_replay_exactly_covers_frozen_inputs_without_rescoring(self):
        projector = ReplayProjector()
        self.assertEqual(len(projector.records), 16)
        for p in evaluation.PLAN['model_inputs']:
            with self.subTest(packet=p):
                r = projector.generate(p)
                self.assertTrue(evaluation.mechanics(r['reply'], p)['valid'])
                self.assertIsNone(r['demo_generation_seconds'])
        self.assertEqual(self.generate()['result']['historical_grade'], 'small_degree_mismatch')

    def test_delivery_requires_review_and_passes_exact_whisper_no_body_or_grade(self):
        c = self.generate()
        self.assertEqual(self.receiver.requests, [])
        with self.assertRaises(DemoError):
            self.send(c, inspected=False)
        self.send(c)
        payload = json.loads(self.receiver.requests[0]['messages'][1]['content'])
        self.assertEqual(payload['body_whisper'], c['text'])
        self.assertEqual(payload['source_ids'], c['source_ids'])
        self.assertIn('replay', payload['origin'])
        self.assertNotIn('sensorium', payload)
        self.assertNotIn('historical_grade', payload)

    def test_other_session_changed_selection_and_forged_token_blocked(self):
        c = self.generate()
        for change in ({'owner': 'browser-b'}, {'voice': 'conversational'},
                       {'case_id': 'focus-final-02'}, {'token': 'forged'}):
            with self.subTest(change=change), self.assertRaises(DemoError):
                self.send(c, **change)
        self.assertEqual(self.receiver.requests, [])

    def test_invalidation_new_generation_and_expiry_block_old_candidates(self):
        c = self.generate()
        self.service.invalidate('browser-a')
        with self.assertRaises(DemoError):
            self.send(c)
        c = self.generate()
        self.generate()
        with self.assertRaises(DemoError):
            self.send(c)
        c = self.generate()
        self.now += 600
        with self.assertRaises(DemoError):
            self.send(c)
        self.assertEqual(self.receiver.requests, [])

    def test_duplicate_and_uncertain_failure_never_retry(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                self.receiver.fail = fail
                c = self.generate()
                if fail:
                    with self.assertRaises(DemoError) as exc:
                        self.send(c)
                    self.assertNotIn('PRIVATE', str(exc.exception))
                else:
                    self.send(c)
                count = len(self.receiver.requests)
                with self.assertRaises(DemoError):
                    self.send(c)
                self.assertEqual(len(self.receiver.requests), count)
        logs = '\n'.join(p.read_text() for p in (self.root / 'events').glob('*.json'))
        self.assertNotIn('PRIVATE', logs)

    def test_bad_output_and_abstention_blocked_even_with_inspection(self):
        p = packet_for('focus-final-01', 'plain')
        base = ReplayProjector().generate(p)
        variants = [dict(base, ended_with_eos=False), dict(base, hit_token_cap=True),
                    dict(base, reply='broken'),
                    dict(base, reply=json.dumps({'case_id': p['case_id'], 'text': '', 'source_ids': []})),
                    dict(base, reply=json.dumps({'case_id': p['case_id'], 'text': 'Text', 'source_ids': ['wrong']}))]
        for result in variants:
            with self.subTest(result=result):
                self.service.projector.generate = lambda p: deepcopy(result)
                c = self.generate()
                self.assertFalse(c['eligible'])
                with self.assertRaises(DemoError):
                    self.send(c)
        self.assertEqual(self.receiver.requests, [])

    def test_receiver_off_never_consumes_or_calls(self):
        self.service.receiver = None
        c = self.generate()
        with self.assertRaises(DemoError):
            self.send(c)
        self.assertFalse(self.service.pending['browser-a']['consumed'])

    def test_invalid_input_cannot_reach_projector(self):
        with self.assertRaises(DemoError):
            self.service.generate('browser-a', 'not-a-case', 'plain')
        self.assertEqual(self.service.pending, {})

    def test_receiver_budget_and_incomplete_response(self):
        calls = []
        data = {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Hello'}}]}
        def send(request, key):
            calls.append(request)
            return deepcopy(data)
        r = NebiusReceiver('synthetic-key', 1, send=send)
        self.assertEqual(r.respond({})['text'], 'Hello')
        with self.assertRaises(DemoError):
            r.respond({})
        self.assertEqual(len(calls), 1)
        for choice in [dict(data['choices'][0], finish_reason='length'),
                       {'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': None, 'reasoning': 'Do not substitute this.'}},
                       {'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Hello', 'tool_calls': [{'id': 'x'}]}}]:
            with self.subTest(choice=choice):
                data['choices'] = [choice]
                r = NebiusReceiver('synthetic-key', 1, send=send)
                with self.assertRaises(DemoError):
                    r.respond({})
                self.assertEqual(r.attempts, 1)

    def test_redirect_rejected(self):
        with self.assertRaises(DemoError):
            NoRedirect().redirect_request(None, None, 302, 'Found', {}, 'https://example.invalid')

    def test_exhausted_limit_is_not_a_dispatch_or_provider_failure(self):
        calls = []
        data = {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Hello'}}]}
        self.service.receiver = NebiusReceiver('synthetic', 1, send=lambda request, key: calls.append(request) or data)
        first = self.generate()
        self.send(first)
        second = self.generate()
        with self.assertRaises(DemoError) as caught:
            self.send(second)
        self.assertEqual(caught.exception.code, 'budget_exhausted')
        self.assertIs(caught.exception.dispatched, False)
        self.assertIn('No request was sent', str(caught.exception))
        self.assertNotIn('billed', str(caught.exception))
        self.assertFalse(self.service.pending['browser-a']['consumed'])
        with self.assertRaises(DemoError) as again:
            self.send(second)
        self.assertEqual(again.exception.code, 'budget_exhausted')
        self.assertEqual(len(calls), 1)
        events = [json.loads(p.read_text()) for p in (self.root / 'events').glob('*.json')]
        self.assertEqual(sum(e['event'] == 'receiver_rejected' for e in events), 2)
        self.assertFalse(any(e['event'] == 'receiver_failed' for e in events))

    def test_duplicate_after_failed_provider_preserves_original_reason(self):
        data = {'choices': [{'finish_reason': 'length', 'message': {'role': 'assistant', 'content': 'partial'}}]}
        self.service.receiver = NebiusReceiver('synthetic', 1, send=lambda r, k: data)
        c = self.generate()
        with self.assertRaises(DemoError) as first:
            self.send(c)
        self.assertEqual(first.exception.code, 'incomplete_response')
        with self.assertRaises(DemoError) as duplicate:
            self.send(c)
        self.assertIn(str(first.exception), str(duplicate.exception))
        self.assertEqual(self.service.receiver.attempts, 1)

    def test_receipt_recovery_reads_legacy_and_new_events_without_other_data(self):
        folder = self.root / 'saved-run' / 'events'
        folder.mkdir(parents=True)
        (folder / 'success.json').write_text(json.dumps({'event': 'receiver_result', 'at': '1',
            'result': {'text': 'saved output', 'finish_reason': 'stop', 'headers': 'SECRET'}}))
        (folder / 'legacy.json').write_text(json.dumps({'event': 'receiver_failed', 'at': '2'}))
        (folder / 'budget.json').write_text(json.dumps({'event': 'receiver_rejected', 'at': '3',
            'code': 'budget_exhausted', 'dispatched': False}))
        (folder / 'broken.json').write_text('{')
        recovered = recover(self.root)
        self.assertEqual(len(recovered['receiver_events']), 3)
        self.assertEqual(recovered['receiver_events'][0]['result']['text'], 'saved output')
        self.assertIn('cannot distinguish', recovered['receiver_events'][1]['note'])
        self.assertFalse(recovered['receiver_events'][2]['dispatched'])
        self.assertEqual(len(recovered['unreadable_files']), 1)
        self.assertNotIn('SECRET', json.dumps(recovered))

    def test_persistent_subprocess_and_timeout_stop_without_restart(self):
        script = self.root / 'fake.py'
        script.write_text('import sys,json,os\nfor line in sys.stdin:\n print(json.dumps({"ok":True,"result":{"pid":os.getpid(),"input":json.loads(line)}}),flush=True)\n')
        worker = LaptopProjector(sys.executable, self.root, self.root / 'worker', timeout=2,
                                 command=[sys.executable, '-u', str(script)])
        self.addCleanup(worker.close)
        p = packet_for('focus-final-01', 'plain')
        a, b = worker.generate(p), worker.generate(p)
        self.assertEqual(a['pid'], b['pid'])
        self.assertEqual(a['input'], {'case_id': p['case_id'], 'voice': p['voice']})
        slow = LaptopProjector(sys.executable, self.root, self.root / 'slow', timeout=.05,
                               command=[sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(slow.close)
        with self.assertRaises(DemoError):
            slow.generate(p)
        self.assertIsNotNone(slow.process.poll())
        with self.assertRaises(DemoError):
            slow.generate(p)

    def test_worker_preserves_virtualenv_interpreter_path(self):
        venv_python = self.root / 'venv-python'
        venv_python.symlink_to(sys.executable)
        worker = LaptopProjector(venv_python, self.root, self.root / 'venv-worker')
        self.assertEqual(worker.command[0], str(venv_python))
        self.assertNotEqual(worker.command[0], str(venv_python.resolve()))
