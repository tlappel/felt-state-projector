"""Two BF16 LoRA updates, adapter save/fresh reload, and in-memory merge check.

Run next to smoke_examples.json in the existing laptop venv. No Hub uploads.
This is a mechanics experiment, not a trained whisper projector.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from uuid import uuid4

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
REVISION = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
FIXTURE_SHA256 = "eced1bdae79a56f90974b1fd0e4c2901f8b153bc6aa2ff670cc8a5e1d23cce66"
MAX_TOKENS = 512
SEED = 1111
GIB = 2**30


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_fixture(path):
    require(digest(path) == FIXTURE_SHA256, "Fixture hash changed; stop.")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["examples"]
    expected = [f"review-{i:02}-1" for i in range(1, 5)]
    require([r["candidate_id"] for r in rows] == expected, "Wrong examples")
    require(len({r["group_id"] for r in rows}) == 4, "Repeated state")
    for row in rows:
        packet, target = row["model_input"], row["target"]
        require(set(packet) == {"case_id", "voice", "sensorium", "source_id"},
                "Unexpected input fields")
        require(set(target) == {"case_id", "text", "source_ids"},
                "Unexpected output fields")
        require(packet["voice"] == "plain", "Unexpected voice")
        require(target["case_id"] == packet["case_id"], "Case ID mismatch")
        require(target["source_ids"] == [packet["source_id"]],
                "Source reference mismatch")
        require(0 < len(target["text"]) <= 160, "Invalid whisper length")
    return data


def masked_sequence(prefix, answer, eos_id, limit=MAX_TOKENS):
    require(bool(prefix) and bool(answer), "Empty prefix or target")
    require(answer[-1] == eos_id, "Target must end with tokenizer EOS")
    require(len(prefix) + len(answer) <= limit,
            "Sequence exceeds cap; no silent truncation is allowed")
    return prefix + answer, [-100] * len(prefix) + answer


def eligible_linear_name(name):
    return name.startswith("backbone.layers.") and not name.endswith(".out_proj")


def encode_examples(tokenizer, fixture):
    require(tokenizer.eos_token_id is not None, "Missing EOS token")
    batches, audit = [], []
    for row in fixture["examples"]:
        packet = json.dumps(row["model_input"], separators=(",", ":"))
        target = json.dumps(row["target"], separators=(",", ":"))
        messages = [
            {"role": "system", "content": fixture["system_prompt"]},
            {"role": "user", "content": packet},
        ]
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=True, enable_thinking=False,
            add_generation_prompt=True,
        )
        # Match generation: append answer tokens to the exact prompt tokens.
        answer = tokenizer.encode(target, add_special_tokens=False)
        require(tokenizer.decode(answer, clean_up_tokenization_spaces=False)
                == target, "Target tokenization did not round-trip")
        answer += [tokenizer.eos_token_id]
        ids, labels = masked_sequence(prefix, answer, tokenizer.eos_token_id)
        batches.append({"ids": ids, "labels": labels, "start": len(prefix)})
        audit.append({
            "candidate_id": row["candidate_id"], "prompt_tokens": len(prefix),
            "answer_tokens_including_eos": len(answer), "total_tokens": len(ids),
            "rendered_prompt": tokenizer.decode(prefix), "target_json": target,
            "input_ids": ids, "labels": labels,
        })
    return batches, audit


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def event(out, stage, **fields):
    record = {"stage": stage, **fields}
    with (out / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def memory(torch):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "free_gib": round(free / GIB, 3),
        "total_gib": round(total / GIB, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3),
    }


def load_base(torch):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, code_revision=REVISION,
        trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": 0}, local_files_only=True,
    )
    require({str(p.device) for p in model.parameters()} == {"cuda:0"},
            "All parameters must be on GPU 0")
    model.config.use_cache = False
    return model


def gpu_batch(torch, batch):
    ids = torch.tensor([batch["ids"]], device="cuda", dtype=torch.long)
    return dict(input_ids=ids, labels=torch.tensor(
        [batch["labels"]], device="cuda", dtype=torch.long),
        attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)


def evaluate(torch, model, batches, tokenizer, out, stage):
    """Teacher-forced comparisons on disclosed training examples, not quality."""
    model.eval()
    references, records = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in batches:
            result = model(**gpu_batch(torch, batch))
            require(torch.isfinite(result.loss).item(), "Nonfinite eval loss")
            logits = result.logits[0, batch["start"] - 1:-1].float().cpu()
            require(torch.isfinite(logits).all().item(), "Nonfinite eval logits")
            predicted = logits.argmax(-1).tolist()
            references.append(logits)
            records.append({"loss": result.loss.item(),
                            "teacher_forced_argmax_ids": predicted,
                            "teacher_forced_text": tokenizer.decode(predicted)})
            del result
    write_json(out / f"{stage}.json", records)
    event(out, stage, mean_training_example_loss=sum(
        r["loss"] for r in records) / len(records))
    return references


def compare_logits(torch, left, right, *, merged):
    metrics = []
    for a, b in zip(left, right, strict=True):
        difference = (a - b).abs()
        metrics.append({
            "max_abs_logit_difference": difference.max().item(),
            "mean_abs_logit_difference": difference.mean().item(),
            "teacher_forced_argmax_agreement":
                (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
            "within_tolerance": torch.allclose(
                a, b, rtol=0.02 if merged else 1e-5,
                atol=0.125 if merged else 1e-5),
        })
        # Merge rounds adapter corrections into BF16 base weights.
        # These numerical smoke tolerances are not semantic acceptance criteria.
    return metrics


def train_and_save(torch, fixture, batches, tokenizer, out):
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    event(out, "loading_base")
    model = load_base(torch)
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and eligible_linear_name(name)]
    require(bool(targets), "No eligible adapter targets")
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
        base_model_name_or_path=MODEL, revision=REVISION,
    ))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    require(bool(trainable) and all(".lora_" in n for n in trainable),
            "Unexpected trainable base parameters")
    require(all(p.dtype == torch.float32 for p in trainable.values()),
            "Expected FP32 adapter parameters with BF16 autocast")
    initial = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    write_json(out / "adapter_inventory.json", {
        "target_modules": targets,
        "trainable_parameters": {n: list(p.shape) for n, p in trainable.items()},
        "count": sum(p.numel() for p in trainable.values()),
    })
    event(out, "adapter_attached", modules=len(targets),
          trainable_parameters=sum(p.numel() for p in trainable.values()))
    before = evaluate(torch, model, batches, tokenizer, out, "before_training")
    del before
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    require(model.is_gradient_checkpointing, "Checkpointing did not activate")
    model.train()
    optimizer = torch.optim.AdamW(
        list(trainable.values()), lr=1e-4, weight_decay=0.0, foreach=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        # Two microbatches of one example per update, four distinct examples total.
        for index in range(step * 2, step * 2 + 2):
            event(out, "microbatch_start", step=step + 1,
                  candidate=fixture["examples"][index]["candidate_id"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(**gpu_batch(torch, batches[index]))
                loss = result.loss
                require(torch.isfinite(loss).item(), "Nonfinite training loss")
                losses.append(loss.item())
                scaled_loss = loss / 2
            scaled_loss.backward()
            del result, loss, scaled_loss
        for name, parameter in trainable.items():
            require(parameter.grad is not None, f"Missing adapter gradient: {name}")
            require(torch.isfinite(parameter.grad).all().item(),
                    f"Nonfinite adapter gradient: {name}")
        require(all(p.grad is None for p in model.parameters() if not p.requires_grad),
                "Frozen base acquired a gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            list(trainable.values()), 1.0, error_if_nonfinite=True)
        require(math.isfinite(norm.item()) and norm.item() > 0,
                "Gradient norm must be finite and nonzero")
        optimizer.step()
        event(out, "optimizer_step", step=step + 1, losses=losses,
              gradient_norm=norm.item(), **memory(torch))
    event(out, "training_finished", seconds=time.perf_counter() - started,
          **memory(torch))
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    require(all(torch.isfinite(p).all().item() for p in trainable.values()),
            "Nonfinite adapter parameter")
    changed = sum(not torch.equal(initial[n], p.detach().cpu())
                  for n, p in trainable.items())
    require(changed > 0, "Optimizer did not change any adapter weights")
    model.gradient_checkpointing_disable()
    model.eval()
    model.save_pretrained(out / "adapter", safe_serialization=True,
                          save_embedding_layers=False)
    tokenizer.save_pretrained(out / "tokenizer")
    saved_state = {n: p.detach().cpu().clone() for n, p in
                   get_peft_model_state_dict(model).items()}
    event(out, "adapter_saved", changed_tensors=changed,
          adapter_sha256=digest(out / "adapter/adapter_model.safetensors"))
    reference = evaluate(torch, model, batches, tokenizer, out, "after_training")
    # No model/tensor aliases escape except CPU reference data.
    return reference, saved_state


def run(out, fixture):
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    from huggingface_hub import try_to_load_from_cache

    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "peft", "accelerate", "mamba-ssm",
        "causal-conv1d", "triton", "numpy", "safetensors", "huggingface-hub")}
    require(versions["torch"] == "2.9.1+cu130", "Unexpected torch version")
    require(versions["transformers"] == "4.48.3", "Unexpected Transformers version")
    require(versions["peft"] == "0.14.0", "This trial pins PEFT 0.14.0")
    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    require(torch.cuda.is_bf16_supported(), "BF16 unavailable")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True)
    batches, audit = encode_examples(tokenizer, fixture)
    write_json(out / "mask_audit.json", audit)
    write_json(out / "smoke_examples.json", fixture)
    hashes = {}
    event(out, "hashing_cached_assets")
    for name in ("config.json", "configuration_nemotron_h.py",
                 "modeling_nemotron_h.py", "model.safetensors",
                 "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        path = try_to_load_from_cache(MODEL, name, revision=REVISION)
        require(isinstance(path, str), f"Missing cached asset: {name}")
        hashes[name] = digest(path)
    write_json(out / "manifest.json", {
        "model": MODEL, "revision": REVISION, "versions": versions,
        "python": sys.version, "gpu": torch.cuda.get_device_name(0),
        "fixture_sha256": FIXTURE_SHA256, "asset_sha256": hashes,
        "script_sha256": digest(__file__), "seed": SEED,
        "steps": 2, "microbatch": 1, "gradient_accumulation": 2,
        "rank": 8, "alpha": 32, "learning_rate": 1e-4,
        "max_tokens": MAX_TOKENS, "compute": "BF16", "adapter_dtype": "FP32",
        "cache": False, "gradient_checkpointing": "non-reentrant",
        "reload_tolerance": {"rtol": 1e-5, "atol": 1e-5},
        "merge_tolerance": {"rtol": 0.02, "atol": 0.125},
    })
    event(out, "data_ready", token_lengths=[len(b["ids"]) for b in batches])
    reference, saved_state = train_and_save(
        torch, fixture, batches, tokenizer, out)
    gc.collect()
    torch.cuda.empty_cache()
    event(out, "loading_fresh_base_for_reload")
    model = PeftModel.from_pretrained(
        load_base(torch), out / "adapter", is_trainable=False)
    reloaded_state = get_peft_model_state_dict(model)
    require(saved_state.keys() == reloaded_state.keys(), "Adapter keys changed")
    for name, value in reloaded_state.items():
        require(torch.equal(saved_state[name], value.detach().cpu()),
                f"Saved adapter tensor mismatch: {name}")
    del reloaded_state, saved_state
    reloaded = evaluate(torch, model, batches, tokenizer, out, "after_reload")
    metrics = compare_logits(torch, reference, reloaded, merged=False)
    write_json(out / "reload_comparison.json", metrics)
    require(all(m["within_tolerance"] for m in metrics),
            "Reload logits differ; inspect reload_comparison.json")
    event(out, "save_reload_passed")
    del reloaded
    event(out, "merging_in_memory")
    model = model.merge_and_unload(safe_merge=True)
    merged = evaluate(torch, model, batches, tokenizer, out, "after_merge")
    metrics = compare_logits(torch, reference, merged, merged=True)
    write_json(out / "merge_comparison.json", metrics)
    require(all(m["within_tolerance"] for m in metrics),
            "Merge logits differ; inspect merge_comparison.json")
    event(out, "merge_numerical_check_passed", comparisons=metrics)
    write_json(out / "result.json", {
        "status": "passed", "optimizer_steps": 2,
        "save_reload": "passed", "in_memory_merge": "numerical_smoke_passed",
        "limits": "Four training states only; no quality or deployment claim."
    })
    print("LoRA smoke passed: two updates, save, fresh reload, in-memory merge.")
    print("This adapter is a mechanics experiment, not a finished projector.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true",
                        help="Validate frozen examples without importing torch")
    args = parser.parse_args()
    fixture = read_fixture(Path(__file__).with_name("smoke_examples.json"))
    if args.check_data:
        print("Frozen four-example fixture passed; no GPU work performed.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("runs") / f"lora-smoke-001-{stamp}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    print("Run directory:", out.resolve(), flush=True)
    try:
        run(out, fixture)
    except BaseException as error:
        (out / "traceback.txt").write_text(traceback.format_exc())
        write_json(out / "result.json", {
            "status": "failed", "error_type": type(error).__name__,
            "error": str(error), "next": "Inspect failure before retrying."
        })
        raise


if __name__ == "__main__":
    main()
