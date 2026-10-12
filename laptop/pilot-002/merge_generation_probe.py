"""Compare eight adapter/merge generations, or run four untuned baseline prompts.

Run beside lora_smoke.py, with the existing training run directory as argument.
The four inputs are disclosed training examples, not final evaluation data.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from uuid import uuid4

from lora_smoke import (
    MODEL, REVISION, digest, encode_examples, event, load_base,
    memory, read_fixture, require, write_json,
)

ADAPTER_SHA256 = "4cc39ef49cbbb54b1f9da9fa98a5ff321923450e4536d2a7241db5e084b2409d"


def unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def mechanics(reply, packet):
    """Check envelope mechanics only; semantic fidelity needs human review."""
    try:
        value = json.loads(reply, object_pairs_hook=unique_keys)
    except (ValueError, TypeError) as error:
        return {"valid": False, "errors": [str(error)]}
    if not isinstance(value, dict):
        return {"valid": False, "errors": ["Expected a JSON object"]}
    errors = []
    if set(value) != {"case_id", "text", "source_ids"}:
        errors.append("Wrong output fields")
    if value.get("case_id") != packet["case_id"]:
        errors.append("Wrong case_id")
    text = value.get("text")
    if not isinstance(text, str) or len(text) > 160:
        errors.append("text must be a string of at most 160 characters")
    expected_sources = [] if text == "" else [packet["source_id"]]
    if value.get("source_ids") != expected_sources:
        errors.append("Wrong source_ids")
    return {"valid": not errors, "errors": errors, "abstention": text == ""}


def run(root, out, base_only=False):
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    from peft import PeftModel

    require(digest(root / "adapter/adapter_model.safetensors") == ADAPTER_SHA256,
            "This diagnostic is pinned to the first saved adapter")
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["model"] == MODEL and manifest["revision"] == REVISION,
            "Unexpected model revision")
    packages = ("torch", "transformers", "peft", "mamba-ssm",
                "causal-conv1d", "triton", "accelerate")
    versions = {name: importlib.metadata.version(name) for name in packages}
    require(all(versions[n] == manifest["versions"][n] for n in packages),
            "Runtime packages changed since training")
    fixture = read_fixture(root / "smoke_examples.json")
    tokenizer = AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True)
    batches, audit = encode_examples(tokenizer, fixture)
    saved_audit = json.loads((root / "mask_audit.json").read_text())
    require(audit == saved_audit, "Tokenizer/prompt audit changed")
    torch.cuda.set_device(0)
    torch.manual_seed(1111)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    config = GenerationConfig(
        max_new_tokens=96, do_sample=False, use_cache=False,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=pad_id,
    )
    write_json(out / "manifest.json", {
        "training_run": str(root.resolve()), "model": MODEL,
        "revision": REVISION, "adapter_sha256": ADAPTER_SHA256,
        "versions": versions, "script_sha256": digest(__file__),
        "prompt_audit_sha256": digest(root / "mask_audit.json"),
        "generation_config": config.to_dict(),
        "scope": ("Four training inputs, untuned baseline, no training" if base_only
                  else "Four training inputs, unmerged vs merged, no training"),
        "adapter_applied": not base_only,
    })
    if base_only:
        event(out, "loading_untuned_base")
        model = load_base(torch)
    else:
        event(out, "loading_saved_adapter")
        model = PeftModel.from_pretrained(
            load_base(torch), root / "adapter", is_trainable=False)
    records = {}
    for phase in (("base",) if base_only else ("unmerged", "merged")):
        if phase == "merged":
            event(out, "merging_in_memory")
            model = model.merge_and_unload(safe_merge=True)
        model.eval()
        records[phase] = []
        torch.cuda.reset_peak_memory_stats()
        for row, batch in zip(fixture["examples"], batches, strict=True):
            event(out, "generation_start", phase=phase,
                  candidate=row["candidate_id"])
            # Only the generation prefix: no target-answer tokens are supplied.
            ids = torch.tensor([batch["ids"][:batch["start"]]],
                               dtype=torch.long, device="cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    generation_config=config,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            generated = output[0, ids.shape[1]:].tolist()
            require(bool(generated), "No generated tokens")
            raw = tokenizer.decode(generated, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
            # Remove only a final EOS for JSON parsing; retain any other tokens.
            ended = generated[-1] == tokenizer.eos_token_id
            content_ids = generated[:-1] if ended else generated
            reply = tokenizer.decode(content_ids, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
            record = {
                "candidate_id": row["candidate_id"], "reply": reply,
                "raw_with_special_tokens": raw, "token_ids": generated,
                "tokens_including_eos": len(generated), "ended_with_eos": ended,
                "hit_token_cap": len(generated) == 96 and not ended,
                "seconds_including_any_compilation": elapsed,
                "mechanics": mechanics(reply, row["model_input"]),
            }
            records[phase].append(record)
            # Save progress after each response, even if a later generation fails.
            write_json(out / f"{phase}.json", records[phase])
            event(out, "response", phase=phase, **record)
            del ids, output
        event(out, "phase_finished", phase=phase, **memory(torch))
    if base_only:
        write_json(out / "result.json", {
            "status": "baseline_completed", "generations": 4,
            "adapter_applied": False, "training_performed": False,
            "semantic_review": "pending",
        })
        print("Untuned baseline complete: four responses, no training.", flush=True)
        print("Report:", out / "base.json", flush=True)
        return
    comparisons = [{
        "candidate_id": a["candidate_id"], "identical_token_ids":
            a["token_ids"] == b["token_ids"],
        "unmerged": a["reply"], "merged": b["reply"],
        "unmerged_mechanics": a["mechanics"],
        "merged_mechanics": b["mechanics"],
    } for a, b in zip(records["unmerged"], records["merged"], strict=True)]
    write_json(out / "comparison.json", comparisons)
    write_json(out / "result.json", {
        "status": "comparison_completed", "generations": 8,
        "semantic_review": "pending", "training_performed": False,
    })
    print("Comparison complete: eight responses, no training.", flush=True)
    print("Report:", out / "comparison.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--base-only", action="store_true",
                        help="Generate four responses from the original base only")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "base-generation" if args.base_only else "merge-generation"
    out = args.run_directory / f"{label}-{stamp}-{uuid4().hex[:8]}"
    out.mkdir(exist_ok=False)
    print("Comparison directory:", out.resolve(), flush=True)
    try:
        run(args.run_directory, out, base_only=args.base_only)
    except BaseException as error:
        (out / "traceback.txt").write_text(traceback.format_exc())
        write_json(out / "result.json", {
            "status": "failed", "error": str(error), "no_automatic_retry": True})
        raise


if __name__ == "__main__":
    main()
