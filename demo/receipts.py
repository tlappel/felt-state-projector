"""Recover saved receiver results without a GPU, credentials or network calls.

Run from the extracted demo folder: python3 demo/receipts.py
Also works as a downloaded standalone file: python3 receipts.py
"""
import argparse
import json
from pathlib import Path


def recover(root):
    rows, unreadable = [], []
    for path in Path(root).glob('*/events/*.json'):
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(row, dict):
                raise ValueError('Invalid event')
        except (OSError, ValueError):
            unreadable.append(str(path))
            continue
        if row.get('event') not in ('receiver_result', 'receiver_failed', 'receiver_rejected'):
            continue
        saved = {key: row[key] for key in ('event', 'at', 'token', 'code', 'status',
                                         'dispatched', 'elapsed_seconds') if key in row}
        saved['file'] = str(path)
        if row['event'] == 'receiver_result':
            result = row.get('result', {})
            saved['result'] = {key: result[key] for key in (
                'text', 'model_reported', 'usage', 'finish_reason', 'request_id', 'attempt')
                if key in result}
        elif row['event'] == 'receiver_failed' and 'code' not in row:
            saved['note'] = ('Older demo recorded no failure reason; this event alone '
                             'cannot distinguish exhausted local budget from provider failure.')
        rows.append(saved)
    rows.sort(key=lambda row: row.get('at', ''))
    return {'receiver_events': rows[-10:], 'unreadable_files': unreadable,
            'note': 'Saved records only. No new request made.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd() / 'demo-runs')
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error('No demo-runs directory here. Run from the extracted demo folder or pass --root.')
    print(json.dumps(recover(args.root), indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
