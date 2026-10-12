"""Isolated laptop worker and bounded Nebius transport."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import urllib.error
import urllib.request

from demo.core import DemoError, ROOT

ENDPOINT = 'https://api.tokenfactory.nebius.com/v1/chat/completions'
MAX_RESPONSE_BYTES = 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DemoError('Provider redirect rejected; no automatic retry.',
                        code='redirect_rejected', dispatched=True)


def send_nebius(request, key):
    req = urllib.request.Request(ENDPOINT, json.dumps(request).encode(), method='POST',
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer ' + key})
    # urllib does not automatically retry requests; redirects are prohibited.
    try:
        response = urllib.request.build_opener(NoRedirect()).open(req, timeout=60)
        with response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise DemoError('Provider response exceeded the capture limit; no automatic retry.',
                            code='response_too_large', dispatched=True)
        return json.loads(raw)
    except DemoError:
        raise
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise DemoError(f'Nebius returned HTTP {status}. No automatic retry.',
                        code='provider_http_error', dispatched=True) from None
    except Exception:
        # Never surface provider bodies, request headers or credential-bearing errors.
        raise DemoError('Provider transport failed. Delivery and billing are uncertain; no automatic retry.',
                        code='transport_failed', dispatched=None) from None


class NebiusReceiver:
    def __init__(self, key, limit, send=send_nebius):
        if not isinstance(key, str) or not key.strip():
            raise DemoError('NEBIUS_API_KEY is missing; receiver was not enabled.')
        if type(limit) is not int or not 1 <= limit <= 100:
            raise DemoError('Choose a receiver attempt limit from 1 to 100.')
        self.key, self.limit, self.send = key, limit, send
        self.attempts = 0
        self.lock = threading.Lock()

    def respond(self, request):
        with self.lock:
            if self.attempts >= self.limit:
                raise DemoError(f'Receiver limit reached ({self.attempts}/{self.limit}). '
                                'No request was sent to Nebius for this click.',
                                code='budget_exhausted', dispatched=False)
            self.attempts += 1
            data = self.send(request, self.key)
            try:
                choices = data['choices']
                if len(choices) != 1:
                    raise ValueError('choice count')
                choice = choices[0]
                message = choice['message']
                content = message.get('content')
                if (choice['finish_reason'] != 'stop' or message.get('role') != 'assistant'
                        or message.get('tool_calls') or not isinstance(content, str)
                        or not content.strip()):
                    raise ValueError('incomplete content')
                content.encode('utf-8', errors='strict')
            except (KeyError, IndexError, TypeError, ValueError, UnicodeError):
                raise DemoError('Nebius returned an incomplete or invalid response. '
                                'This attempt may have been billed; no automatic retry.',
                                code='incomplete_response', dispatched=True) from None
            return {'text': content, 'model_reported': data.get('model'),
                    'usage': data.get('usage'), 'finish_reason': choice['finish_reason'],
                    'request_id': data.get('id'), 'attempt': self.attempts}


class LaptopProjector:
    mode = 'Live saved adapter — laptop / configured GPU host'

    def __init__(self, python, run_dir, log_dir, *, timeout=600, command=None):
        # Preserve the venv symlink path: resolving it selects the system Python.
        self.command = command or [str(Path(python).absolute()), '-u',
                                   str(ROOT / 'demo/worker.py'), str(Path(run_dir).resolve())]
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.process = None
        self.failed = False
        self.lock = threading.Lock()
        self.lines = queue.Queue(maxsize=2)
        self.log_file = None

    def _read(self):
        try:
            while True:
                line = self.process.stdout.readline(MAX_RESPONSE_BYTES + 1)
                if not line or len(line) > MAX_RESPONSE_BYTES:
                    self.lines.put(None)
                    return
                self.lines.put(line)
        except (OSError, ValueError):
            self.lines.put(None)

    def _start(self):
        env = dict(os.environ)
        env.pop('NEBIUS_API_KEY', None)
        env['PYTHONUNBUFFERED'] = '1'
        self.log_file = (self.log_dir / 'worker.log').open('a', encoding='utf-8')
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log_file,
                                        cwd=ROOT, env=env, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()

    def generate(self, packet):
        with self.lock:
            if self.failed:
                raise DemoError('Worker stopped; restart the demo to reload it.')
            try:
                if self.process is None:
                    self._start()
                self.process.stdin.write(json.dumps({'case_id': packet['case_id'],
                                                     'voice': packet['voice']}) + '\n')
                self.process.stdin.flush()
                line = self.lines.get(timeout=self.timeout)
                if line is None:
                    raise DemoError('Worker exited or exceeded its response limit.')
                reply = json.loads(line)
                if not reply.get('ok'):
                    raise DemoError('Worker failed; inspect worker.log.')
                return reply['result']
            except Exception:
                self.failed = True
                self.close()
                raise DemoError('Worker failed or timed out; no restart or retry was attempted.') from None

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
        if self.log_file:
            self.log_file.close()
