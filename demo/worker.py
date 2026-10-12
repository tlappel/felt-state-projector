"""Run ONLY in the existing training venv. JSON-lines IPC; diagnostics on stderr."""
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from demo.core import evaluation as e
    root = Path(sys.argv[1]).resolve()
    # Keep the protocol stream clean even when third-party libraries print.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    started = time.perf_counter()
    data, _, _ = e.validate_saved_run(root)
    import torch
    from peft import PeftModel
    from transformers import GenerationConfig
    e.require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.cuda.set_device(0)
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)
    tokenizer, prompts = e.validate_tokenizer(root, data)
    base = e.load_base(torch)
    # Save the target audit under a new demo directory, never in the training run.
    from uuid import uuid4
    audit = Path(__file__).resolve().parents[1] / 'demo-runs' / ('worker-' + uuid4().hex)
    audit.mkdir(parents=True)
    e.verify_base_targets(torch, base, root, audit, 'demo')
    base.requires_grad_(False)
    model = PeftModel.from_pretrained(base, root / 'best_adapter', is_trainable=False)
    e.verify_loaded_adapter(model, root)
    model.eval()
    e.require(not any(p.requires_grad for p in model.parameters()), 'Parameters must be frozen')
    config = GenerationConfig(max_new_tokens=96, do_sample=False, use_cache=False,
                              eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id)
    startup = time.perf_counter() - started
    first = True
    for line in sys.stdin:
        try:
            request = json.loads(line)
            row = next(p for p in prompts if (p['model_input']['case_id'], p['model_input']['voice'])
                       == (request['case_id'], request['voice']))
            result = e.generate_one(torch, model, tokenizer, row, config)
            result.update(startup_seconds=startup if first else 0.0,
                          demo_generation_seconds=result['seconds_including_any_compilation'])
            first = False
            protocol.write(json.dumps({'ok': True, 'result': result}) + '\n')
            protocol.flush()
        except Exception:
            traceback.print_exc(file=sys.stderr)
            protocol.write(json.dumps({'ok': False}) + '\n')
            protocol.flush()
            return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
