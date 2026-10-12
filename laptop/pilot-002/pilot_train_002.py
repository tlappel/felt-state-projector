"""Bounded second quality pilot: 92 training / 24 development examples.

Run next to pilot_examples_002.json, the original lora_smoke.py, and
merge_generation_probe.py. No new dependencies, model downloads or uploads.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time
import traceback
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from uuid import uuid4

from lora_smoke import (
    MODEL, REVISION, digest, eligible_linear_name, encode_examples,
    event, gpu_batch, load_base, memory, require, write_json,
)
from merge_generation_probe import mechanics

DATA_SHA256 = "1fbf6c65c404d3b5307a7e9eed03b90494ee4b7d38f61e1289095ec26f4baeba"
SEED = 1111
MAX_EPOCHS = 8
MIN_EPOCHS = 3
PATIENCE = 2
MIN_DELTA = 0.002
ACCUMULATION = 4
LEARNING_RATE = 2e-5

HELPER_SHA256 = {
    "lora_smoke.py": "6ffc8affc90aab19ce81932e328b0fcce629034190cf941490bcaa7623e16e57",
    "merge_generation_probe.py": "52bf2af3bc2bb122be35bf9d69db814f858e8ad807078c4301611977b017023b",
}
REPORT_FILES = (
    "result.json", "manifest.json", "dataset.json", "learning_curve.json",
    "before_training_responses.json", "after_training_responses.json",
    "adapter_inventory.json", "train_mask_audit.json", "development_mask_audit.json",
    "events.jsonl", "traceback.txt", "pilot_train_002.py", "lora_smoke.py",
    "merge_generation_probe.py",
)


def check_helpers():
    for name, expected in HELPER_SHA256.items():
        require(digest(Path(__file__).with_name(name)) == expected,
                f"Bundled helper changed: {name}")


def package_report(out):
    # Explicit allowlist: neither model weights nor unrelated laptop files leave the run.
    members = [name for name in REPORT_FILES if (out / name).is_file()]
    require("result.json" in members, "No result to package")
    hashes = {name: digest(out / name) for name in members}
    write_json(out / "report_hashes.json", hashes)
    path = out / "pilot_002_report.zip"
    with ZipFile(path, "x", compression=ZIP_DEFLATED) as archive:
        for name in members + ["report_hashes.json"]:
            archive.write(out / name, name)
    print("Report:", path.resolve(), flush=True)
    return path


def development_subgroups(rows, per_state):
    groups = {"zero_energy": set(), "nonzero_energy": set(),
              "original_six_states": set(), "new_four_states": set()}
    for row in rows:
        body = row["model_input"]["sensorium"]["body"]
        groups["zero_energy" if body["energy"] == 0 else "nonzero_energy"].add(row["group_id"])
        groups["new_four_states" if row["candidate_id"].startswith("focus-dev-")
               else "original_six_states"].add(row["group_id"])
    return {name: {"states": len(ids), "mean_loss":
                   sum(per_state[g] for g in ids) / len(ids) if ids else None}
            for name, ids in groups.items()}


def read_data(path):
    require(digest(path) == DATA_SHA256, "Pilot dataset changed; stop")
    data = json.loads(path.read_text(encoding="utf-8"))
    train, dev = data["train"], data["development"]
    require(len(train) == 92 and len(dev) == 24, "Wrong split sizes")
    tg, dg = {r["group_id"] for r in train}, {r["group_id"] for r in dev}
    require(len(tg) == 40 and len(dg) == 10 and not tg & dg,
            "Body-state split leakage")
    require(len({r["candidate_id"] for r in train + dev}) == 116,
            "Duplicate candidate IDs")
    require(not {r["candidate_id"] for r in train + dev}
            & set(data["excluded_tentative_candidates"]), "Tentative target included")
    for row in train + dev:
        packet, target = row["model_input"], row["target"]
        require(set(packet) == {"case_id", "voice", "sensorium", "source_id"},
                "Unexpected input fields")
        require(set(target) == {"case_id", "text", "source_ids"},
                "Unexpected target fields")
        require(target["case_id"] == packet["case_id"], "Case mismatch")
        require(target["source_ids"] == [packet["source_id"]], "Source mismatch")
        require(0 < len(target["text"]) <= 160, "Invalid target length")
    return data


def generation_indices(rows):
    seen, selected = set(), []
    for index, row in enumerate(rows):
        key = (row["group_id"], row["model_input"]["voice"])
        if key not in seen:
            selected.append(index)
            seen.add(key)
    return selected


def development_loss(torch, model, rows, batches):
    model.eval()
    groups = defaultdict(list)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row, batch in zip(rows, batches, strict=True):
            result = model(**gpu_batch(torch, batch))
            loss = result.loss.item()
            require(math.isfinite(loss), "Nonfinite development loss")
            groups[row["group_id"]].append(loss)
            del result
    per_state = {g: sum(values) / len(values) for g, values in groups.items()}
    return sum(per_state.values()) / len(per_state), per_state


def generate_development(torch, model, tokenizer, rows, batches, out, stage):
    from transformers import GenerationConfig
    model.eval()
    pad = tokenizer.pad_token_id
    config = GenerationConfig(
        max_new_tokens=96, do_sample=False, use_cache=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id if pad is None else pad,
    )
    records = []
    for index in generation_indices(rows):
        row, batch = rows[index], batches[index]
        event(out, "generation_start", phase=stage, candidate=row["candidate_id"])
        ids = torch.tensor([batch["ids"][:batch["start"]]],
                           dtype=torch.long, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                generation_config=config)
        tokens = output[0, ids.shape[1]:].tolist()
        require(bool(tokens), "No generated tokens")
        ended = tokens[-1] == tokenizer.eos_token_id
        reply = tokenizer.decode(tokens[:-1] if ended else tokens,
                                 skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
        record = {
            "candidate_id": row["candidate_id"],
            "voice": row["model_input"]["voice"], "reply": reply,
            "token_ids": tokens, "ended_with_eos": ended,
            "hit_token_cap": len(tokens) == 96 and not ended,
            "mechanics": mechanics(reply, row["model_input"]),
        }
        records.append(record)
        write_json(out / f"{stage}_responses.json", records)
        event(out, "response", phase=stage, candidate=record["candidate_id"],
              voice=record["voice"], reply=reply, mechanics=record["mechanics"])
        del ids, output
    return records


def fit(torch, model, tokenizer, data, train_batches, dev_batches, out):
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    require(trainable and all(".lora_" in n for n in trainable),
            "Only adapter parameters may be trained")
    require(all(p.dtype == torch.float32 for p in trainable.values()),
            "Expected FP32 adapters and BF16 autocast")
    baseline, per_state = development_loss(
        torch, model, data["development"], dev_batches)
    best_loss, best_epoch, patience_loss, stale = baseline, 0, baseline, 0
    model.save_pretrained(out / "best_adapter", safe_serialization=True,
                          save_embedding_layers=False)
    history = [{"epoch": 0, "development_loss": baseline,
                "per_state": per_state, "best_epoch": 0,
                "subgroups": development_subgroups(data["development"], per_state)}]
    write_json(out / "learning_curve.json", history)
    event(out, "baseline_development_loss", loss=baseline)
    # LoRA B starts at zero, so this is the original-base behavior.
    generate_development(torch, model, tokenizer, data["development"],
                         dev_batches, out, "before_training")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    require(model.is_gradient_checkpointing, "Checkpointing did not activate")
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=LEARNING_RATE,
                                  weight_decay=0.0, foreach=False)
    rng = random.Random(SEED)
    steps = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, MAX_EPOCHS + 1):
        order = list(range(len(train_batches)))
        rng.shuffle(order)
        model.train()
        losses = []
        event(out, "epoch_start", epoch=epoch, max_epochs=MAX_EPOCHS)
        for offset in range(0, len(order), ACCUMULATION):
            indices = order[offset:offset + ACCUMULATION]
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(**gpu_batch(torch, train_batches[index]))
                    loss = result.loss
                    require(torch.isfinite(loss).item(), "Nonfinite training loss")
                    losses.append(loss.item())
                    scaled = loss / len(indices)
                scaled.backward()
                del result, loss, scaled
            require(all(p.grad is not None for p in trainable.values()),
                    "An adapter parameter has no gradient")
            require(all(p.grad is None for p in model.parameters()
                        if not p.requires_grad), "Frozen base acquired gradients")
            norm = torch.nn.utils.clip_grad_norm_(
                list(trainable.values()), 1.0, error_if_nonfinite=True)
            require(norm.item() > 0, "All adapter gradients are zero")
            optimizer.step()
            steps += 1
            event(out, "optimizer_step", epoch=epoch, step=steps,
                  mean_microbatch_loss=sum(losses[-len(indices):]) / len(indices),
                  gradient_norm=norm.item())
        optimizer.zero_grad(set_to_none=True)
        score, per_state = development_loss(
            torch, model, data["development"], dev_batches)
        require(all(torch.isfinite(p).all().item() for p in trainable.values()),
                "Nonfinite adapter parameters")
        if score < best_loss:
            best_loss, best_epoch = score, epoch
            model.save_pretrained(out / "best_adapter", safe_serialization=True,
                                  save_embedding_layers=False)
        if score < patience_loss - MIN_DELTA:
            patience_loss, stale = score, 0
        else:
            stale += 1
        record = {"epoch": epoch, "optimizer_steps": steps,
                  "training_loss": sum(losses) / len(losses),
                  "development_loss": score, "per_state": per_state,
                  "subgroups": development_subgroups(data["development"], per_state),
                  "best_epoch": best_epoch, "epochs_without_material_gain": stale,
                  **memory(torch)}
        history.append(record)
        write_json(out / "learning_curve.json", history)
        event(out, "epoch_finished", **record)
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            event(out, "early_stop", reason="Development loss stopped improving")
            break
    model.gradient_checkpointing_disable()
    event(out, "training_finished", best_epoch=best_epoch, best_loss=best_loss,
          optimizer_steps=steps, seconds=time.perf_counter() - started,
          **memory(torch))
    return best_epoch, best_loss, steps


def run(out, data):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer
    versions = {n: importlib.metadata.version(n) for n in (
        "torch", "transformers", "peft", "accelerate", "mamba-ssm",
        "causal-conv1d", "triton", "safetensors", "huggingface-hub")}
    require(versions["torch"] == "2.9.1+cu130"
            and versions["transformers"] == "4.48.3"
            and versions["peft"] == "0.14.0", "Unexpected core package versions")
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(torch.cuda.is_bf16_supported(), "BF16 unavailable")
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True)
    batches = {}
    for split in ("train", "development"):
        batches[split], audit = encode_examples(tokenizer, {
            "system_prompt": data["system_prompt"], "examples": data[split]})
        write_json(out / f"{split}_mask_audit.json", audit)
        event(out, "data_ready", split=split, examples=len(audit),
              maximum_tokens=max(len(b["ids"]) for b in batches[split]))
    write_json(out / "dataset.json", data)
    tokenizer.save_pretrained(out / "tokenizer")
    write_json(out / "manifest.json", {
        "model": MODEL, "revision": REVISION, "versions": versions,
        "gpu": torch.cuda.get_device_name(0), "seed": SEED,
        "dataset_sha256": DATA_SHA256, "script_sha256": digest(__file__),
        "source_helpers_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("lora_smoke.py", "merge_generation_probe.py")},
        "rank": 8, "alpha": 32, "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS, "min_epochs": MIN_EPOCHS,
        "patience": PATIENCE, "min_delta": MIN_DELTA,
        "microbatch": 1, "accumulation": ACCUMULATION,
        "maximum_updates": MAX_EPOCHS * math.ceil(len(data["train"]) / ACCUMULATION),
        "compute": "BF16", "adapter_parameters": "FP32", "cache": False,
        "checkpoint_selection": "lowest development loss, including epoch zero",
        "development_loss_weighting": "equal state weight; mean variants per state",
        "free_generations": 2 * len(generation_indices(data["development"])),
        "max_new_tokens_per_generation": 96,
    })
    event(out, "loading_fresh_original_base")
    model = load_base(torch)
    targets = [n for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and eligible_linear_name(n)]
    require(len(targets) == 71, "Expected the same 71 adapter target layers")
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
        base_model_name_or_path=MODEL, revision=REVISION))
    require(all(torch.count_nonzero(p).item() == 0 for n, p in
                model.named_parameters() if ".lora_B." in n),
            "Initial adapter must contribute zero")
    inventory = {n: list(p.shape) for n, p in model.named_parameters() if p.requires_grad}
    require(len(inventory) == 142, "Expected 142 trainable adapter tensors")
    require(sum(p.numel() for p in model.parameters() if p.requires_grad) == 8527104,
            "Unexpected trainable parameter count")
    write_json(out / "adapter_inventory.json", {
        "target_modules": targets, "trainable_parameters": inventory})
    best_epoch, best_loss, steps = fit(
        torch, model, tokenizer, data, batches["train"], batches["development"], out)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    event(out, "reloading_best_adapter", epoch=best_epoch)
    model = PeftModel.from_pretrained(
        load_base(torch), out / "best_adapter", is_trainable=False)
    reloaded_loss, _ = development_loss(
        torch, model, data["development"], batches["development"])
    require(abs(reloaded_loss - best_loss) <= 1e-5,
            "Best-adapter reload loss differs; inspect before retrying")
    generate_development(torch, model, tokenizer, data["development"],
                         batches["development"], out, "after_training")
    summary = {
        "dataset_sha256": DATA_SHA256, "pilot": "pilot-002",
        "status": "pilot_completed_review_pending", "best_epoch": best_epoch,
        "best_development_loss": best_loss, "reloaded_loss": reloaded_loss,
        "optimizer_steps": steps, "adapter_sha256":
            digest(out / "best_adapter/adapter_model.safetensors"),
        "next": "Review development responses; final evaluation untouched",
    }
    write_json(out / "result.json", summary)
    event(out, "pilot_finished", **summary)
    print("Send pilot_002_report.zip from the printed run directory.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    check_helpers()
    data = read_data(Path(__file__).with_name("pilot_examples_002.json"))
    if args.check_data:
        print("Frozen split passed: 92 train / 24 development; 40 / 10 states. No GPU run.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("runs") / f"pilot-002-{stamp}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    print("Pilot directory:", out.resolve(), flush=True)
    try:
        for name in ("pilot_train_002.py", *HELPER_SHA256):
            shutil.copyfile(Path(__file__).with_name(name), out / name)
        run(out, data)
    except BaseException as error:
        (out / "traceback.txt").write_text(traceback.format_exc())
        write_json(out / "result.json", {"status": "failed", "error": str(error),
                                         "no_automatic_retry": True})
        try:
            package_report(out)
        except Exception as packaging_error:
            print("Failure report packaging also failed:", packaging_error, flush=True)
        raise
    package_report(out)


if __name__ == "__main__":
    main()
