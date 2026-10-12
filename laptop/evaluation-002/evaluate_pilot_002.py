"""Reserved comparison: 16 untuned + 16 saved-adapter responses; no training.

Copy the evaluation-002 folder into the existing laptop project.
python -u evaluate_pilot_002.py
Optional explicit path: python -u evaluate_pilot_002.py runs/pilot-002-YOUR-RUN
--check-only verifies saved artifacts/packages without loading a model.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time
import traceback
from uuid import uuid4
import zipfile

from lora_smoke import (
    MODEL, REVISION, digest, encode_examples, event, load_base, memory, require,
    write_json,
)

# Frozen before any reserved response is generated. No target-answer prose here.
PLAN = json.loads(Path(__file__).with_name('evaluation_plan.json').read_text(encoding='utf-8'))
PLAN_SHA256 = '574942b113560bd2fcdffd69081d61e0b6d0d7a8f3b03003997e98bc0ed7807a'
DATA_CANONICAL_SHA256 = PLAN['data_canonical_sha256']
RESULT_CANONICAL_SHA256 = PLAN['result_canonical_sha256']
ADAPTER_SHA256 = PLAN['candidate_adapter_sha256']
ARMS = ('untuned_raw', 'saved_pilot_adapter_raw')
MAX_NEW_TOKENS = 96


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_package():
    require(canonical_sha(PLAN) == PLAN_SHA256, 'Evaluation plan changed')
    package = Path(__file__).resolve().parent
    require(digest(package / 'lora_smoke.py') == PLAN['helper_sha256'],
            'Packaged training helper changed')
    require(digest(package / PLAN['review_protocol']['rubric']) ==
            PLAN['review_protocol']['rubric_sha256'], 'Grading document changed')


def resolve_targets(selectors, module_names):
    # PEFT 0.14 list/set matching: exact name OR a dot-delimited suffix.
    return {name for name in module_names
            if any(name == selector or name.endswith('.' + selector)
                   for selector in selectors)}


def validate_target_selectors(config, inventory):
    selectors = config.get('target_modules')
    expected = inventory.get('target_modules')
    require(isinstance(selectors, list) and selectors
            and all(isinstance(s, str) and s for s in selectors),
            'Expected a nonempty target selector list')
    require(isinstance(expected, list) and len(expected) == 71
            and all(isinstance(s, str) and s for s in expected)
            and len(set(expected)) == 71,
            'Training inventory must contain 71 distinct full target names')
    require(all(name.startswith('backbone.layers.') and not name.endswith('.out_proj')
                for name in expected), 'Unsupported inventory target')
    require(not config.get('exclude_modules')
            and config.get('layers_to_transform') in (None, [])
            and config.get('layers_pattern') in (None, [])
            and not config.get('layer_replication'), 'Unexpected target filtering or layer replication')
    missing = set(expected) - resolve_targets(selectors, expected)
    unused = [s for s in selectors if not resolve_targets([s], expected)]
    require(not missing and not unused,
            f'Adapter selector mismatch: missing={sorted(missing)}, unused={sorted(unused)}')
    return {'saved_selectors': sorted(selectors), 'inventory_modules': sorted(expected),
            'saved_selector_count': len(selectors), 'expected_module_count': len(expected)}


def verify_base_targets(torch, model, root, out, arm):
    config = read_json(root / 'best_adapter/adapter_config.json')
    inventory = read_json(root / 'adapter_inventory.json')
    audit = validate_target_selectors(config, inventory)
    modules = dict(model.named_modules())
    resolved = resolve_targets(config['target_modules'], modules)
    expected = set(inventory['target_modules'])
    audit.update({'resolved_modules': sorted(resolved),
                  'missing_modules': sorted(expected - resolved),
                  'extra_modules': sorted(resolved - expected)})
    write_json(out / f'target_resolution_{arm}.json', audit)
    require(resolved == expected,
            f'Actual model target mismatch; inspect target_resolution_{arm}.json')
    require(all(isinstance(modules[name], torch.nn.Linear) for name in resolved),
            'A resolved adapter target is not a Linear layer')


def verify_loaded_adapter(model, root):
    inventory = read_json(root / 'adapter_inventory.json')
    require(set(model.base_model.targeted_module_names) == set(inventory['target_modules']),
            'PEFT attached adapters to unexpected modules')
    actual = {name: list(parameter.shape) for name, parameter in model.named_parameters()
              if '.lora_' in name}
    require(actual == inventory['trainable_parameters'],
            'Loaded adapter tensor names/shapes differ from training inventory')


def discover_run(explicit=None):
    if explicit is not None:
        return explicit.resolve()
    candidates = []
    for path in sorted(Path('runs').glob('pilot-002-*/result.json')):
        value = read_json(path)
        if value.get('adapter_sha256') == ADAPTER_SHA256:
            candidates.append(path.parent.resolve())
    require(len(candidates) == 1,
            'Expected one matching pilot under ./runs; found '
            + str(len(candidates)) + '. Pass the pilot run directory explicitly. '
            + 'Matches: ' + ', '.join(map(str, candidates)))
    return candidates[0]


def validate_saved_run(root):
    validate_package()
    result = read_json(root / 'result.json')
    require(canonical_sha(result) == RESULT_CANONICAL_SHA256,
            'Pilot result does not match the reviewed run')
    data = read_json(root / 'dataset.json')
    require(canonical_sha(data) == DATA_CANONICAL_SHA256, 'Pilot dataset changed')
    require(data['system_prompt'] == PLAN['system_prompt'], 'System prompt changed')
    for name, expected in PLAN['training_artifact_sha256'].items():
        require(digest(root / name) == expected, 'Reviewed training artifact changed: ' + name)
    adapter = root / 'best_adapter'
    require(digest(adapter / 'adapter_model.safetensors') == ADAPTER_SHA256,
            'Wrong adapter weights; expected the reviewed epoch-7 pilot')
    manifest = read_json(root / 'manifest.json')
    require(manifest['model'] == MODEL == PLAN['model']
            and manifest['revision'] == REVISION == PLAN['revision'],
            'Wrong base checkpoint or revision')
    for field, expected in {'rank': 8, 'alpha': 32, 'compute': 'BF16',
                            'adapter_parameters': 'FP32', 'cache': False}.items():
        require(manifest.get(field) == expected, 'Wrong pilot manifest field: ' + field)
    config = read_json(adapter / 'adapter_config.json')
    for field, expected in {
        'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 8,
        'lora_alpha': 32, 'lora_dropout': 0.0, 'bias': 'none',
        'base_model_name_or_path': MODEL, 'revision': REVISION,
    }.items():
        require(config.get(field) == expected, 'Wrong adapter config field: ' + field)
    require(not config.get('use_dora', False) and not config.get('use_rslora', False)
            and not config.get('modules_to_save') and not config.get('rank_pattern')
            and not config.get('alpha_pattern'), 'Unexpected adapter configuration')
    inventory = read_json(root / 'adapter_inventory.json')
    target_audit = validate_target_selectors(config, inventory)
    print('Adapter target selectors:', target_audit['saved_selector_count'],
          '| Training layers:', target_audit['expected_module_count'], flush=True)
    helper = Path(__file__).with_name('lora_smoke.py')
    require(digest(helper) == manifest['source_helpers_sha256']['lora_smoke.py'],
            'lora_smoke.py differs from the helper used for training')
    versions = {name: importlib.metadata.version(name) for name in manifest['versions']}
    require(versions == manifest['versions'], 'Installed package versions changed since training')
    hashes = {str(p.relative_to(root)): digest(p) for p in [
        root / 'result.json', root / 'dataset.json', root / 'manifest.json',
        root / 'adapter_inventory.json', adapter / 'adapter_config.json',
        adapter / 'adapter_model.safetensors', root / 'train_mask_audit.json',
        root / 'development_mask_audit.json']}
    tokenizer_files = sorted(p for p in (root / 'tokenizer').rglob('*') if p.is_file())
    require(bool(tokenizer_files), 'Saved tokenizer directory is empty')
    hashes.update({str(p.relative_to(root)): digest(p) for p in tokenizer_files})
    return data, versions, hashes


def messages_for(packet):
    require(set(packet) == {'case_id', 'voice', 'sensorium', 'source_id'},
            'Review metadata must not enter the model input')
    return [{'role': 'system', 'content': PLAN['system_prompt']},
            {'role': 'user', 'content': json.dumps(packet, separators=(',', ':'))}]


def prepare_prompts(tokenizer):
    require(tokenizer.eos_token_id is not None, 'Missing EOS token')
    rows = []
    for packet in PLAN['model_inputs']:
        ids = tokenizer.apply_chat_template(
            messages_for(packet), tokenize=True, enable_thinking=False,
            add_generation_prompt=True)
        require(bool(ids) and len(ids) + MAX_NEW_TOKENS <= 512,
                'Prompt plus generation allowance exceeds the bounded context')
        rows.append({'model_input': packet, 'input_ids': ids,
                     'rendered_prompt': tokenizer.decode(ids)})
    require(len(rows) == 16, 'Wrong prompt count')
    return rows


def validate_tokenizer(root, data):
    from transformers import AutoTokenizer
    saved = AutoTokenizer.from_pretrained(root / 'tokenizer', local_files_only=True)
    pinned = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    for split in ('train', 'development'):
        fixture = {'system_prompt': data['system_prompt'], 'examples': data[split]}
        _, audit = encode_examples(saved, fixture)
        _, pinned_audit = encode_examples(pinned, fixture)
        require(audit == read_json(root / f'{split}_mask_audit.json') == pinned_audit,
                'Saved/pinned tokenizer differs from the training token audit')
    require((saved.eos_token_id, saved.pad_token_id) ==
            (pinned.eos_token_id, pinned.pad_token_id), 'Special token IDs changed')
    prompts = prepare_prompts(saved)
    require(prompts == prepare_prompts(pinned), 'Reserved prompt tokenizations differ')
    return saved, prompts


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def mechanics(reply, packet):
    try:
        obj = json.loads(reply, object_pairs_hook=unique_keys)
    except (ValueError, TypeError) as exc:
        return {'valid': False, 'errors': [str(exc)], 'abstention': False}
    if not isinstance(obj, dict):
        return {'valid': False, 'errors': ['Expected a JSON object'], 'abstention': False}
    errors = []
    if set(obj) != {'case_id', 'text', 'source_ids'}:
        errors.append('Wrong fields')
    if obj.get('case_id') != packet['case_id']:
        errors.append('Wrong case_id')
    text = obj.get('text')
    if not isinstance(text, str) or len(text) > 160 or (text != '' and not text.strip()):
        errors.append('Expected nonblank text <=160 codepoints, or empty abstention')
    elif isinstance(text, str):
        try:
            text.encode('utf-8', errors='strict')
        except UnicodeError:
            errors.append('Invalid Unicode in text')
    abstention = text == ''
    if obj.get('source_ids') != ([] if abstention else [packet['source_id']]):
        errors.append('Wrong source_ids')
    return {'valid': not errors, 'errors': errors, 'abstention': abstention}


def response_record(tokenizer, tokens, packet, elapsed):
    ended = bool(tokens) and tokens[-1] == tokenizer.eos_token_id
    reply = tokenizer.decode(tokens[:-1] if ended else tokens,
                             skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {'case_id': packet['case_id'], 'voice': packet['voice'],
            'reply': reply, 'raw_with_special_tokens': tokenizer.decode(
                tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            'token_ids': tokens, 'tokens_including_eos': len(tokens),
            'ended_with_eos': ended, 'hit_token_cap': len(tokens) == MAX_NEW_TOKENS and not ended,
            'generation_complete': ended, 'seconds_including_any_compilation': elapsed,
            'mechanics': mechanics(reply, packet)}


def generate_one(torch, model, tokenizer, row, config):
    ids = torch.tensor([row['input_ids']], dtype=torch.long, device='cuda')
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                generation_config=config)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tokens = output[0, ids.shape[1]:].tolist()
    require(bool(tokens), 'Model returned no tokens')
    return response_record(tokenizer, tokens, row['model_input'], elapsed)


def review_packet(records):
    cases = {r['case_id']: r for r in PLAN['cases_for_review_only']}
    pairs = [(arm, row) for arm in ARMS for row in records[arm]]
    random.SystemRandom().shuffle(pairs)
    sheet, key = [], []
    for i, (arm, row) in enumerate(pairs, 1):
        sample = f'response-{i:03}'
        case = cases[row['case_id']]
        packet = next(p for p in PLAN['model_inputs']
                      if p['case_id'] == row['case_id'] and p['voice'] == row['voice'])
        sheet.append({'sample_id': sample, 'model_input': packet,
                      'family': case['family'], 'review_only_categories': case['review_only_categories'],
                      'reply': row['reply'], 'mechanics': row['mechanics'],
                      'ended_with_eos': row['ended_with_eos'], 'hit_token_cap': row['hit_token_cap'],
                      'reviewer': None, 'fidelity': {field: None for field in (
                          'coverage_and_independence', 'evidence_support', 'agency_boundary')},
                      'expression': {'naturalness': None, 'requested_voice_fit': None},
                      'intensity_by_field': {field: None for field in ('hunger', 'energy', 'fatigue')},
                      'meaning_grade': None, 'first_person_scope': None,
                      'findings': [],
                      'notes': None})
        key.append({'sample_id': sample, 'arm': arm,
                    'case_id': row['case_id'], 'voice': row['voice']})
    return sheet, key


def complete_outputs(out, records):
    for arm in ARMS:
        require(len(records[arm]) == 16, 'Incomplete phase: ' + arm)
        require({(r['case_id'], r['voice']) for r in records[arm]} ==
                {(p['case_id'], p['voice']) for p in PLAN['model_inputs']}, 'Response coverage changed')
    sheet, key = review_packet(records)
    write_json(out / 'review_sheet.json', sheet)
    write_json(out / 'review_key.json', key)
    cases = {r['case_id']: r for r in PLAN['cases_for_review_only']}
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for family in ('zero_energy_combination', 'nonzero_energy_combination'):
            rows = [r for r in records[arm] if cases[r['case_id']]['family'] == family]
            summary[arm][family] = {
                'responses': len(rows), 'states': len({r['case_id'] for r in rows}),
                'mechanics_valid': sum(r['mechanics']['valid'] for r in rows),
                'abstentions': sum(r['mechanics']['abstention'] for r in rows),
                'completed_with_eos': sum(r['ended_with_eos'] for r in rows),
                'hit_token_cap': sum(r['hit_token_cap'] for r in rows)}
    write_json(out / 'result.json', {
        'status': 'generation_completed_semantic_review_pending',
        'generations': 32, 'states': 8, 'training_performed': False,
        'adapter_sha256': ADAPTER_SHA256, 'plan_sha256': PLAN_SHA256,
        'mechanical_summary': summary,
        'review_order': 'Read review_sheet.json before review_key.json or arm response files.',
        'next': 'Human rubric review; no automatic retry, retraining or broad reliability claim.'})


def package_report(out):
    # Explicit allowlist: never include weights, credentials or unrelated run files.
    allowed = ['result.json', 'manifest.json', 'execution_plan.json', 'prompt_audit.json',
               'review_sheet.json', 'review_key.json', 'events.jsonl', 'traceback.txt',
               'evaluation-rubric-0.2.md', 'evaluate_pilot_002.py', 'lora_smoke.py']
    allowed += [arm + '.json' for arm in ARMS]
    allowed += [f'target_resolution_{arm}.json' for arm in ARMS]
    present = [name for name in allowed if (out / name).is_file()]
    write_json(out / 'report_hashes.json', {name: digest(out / name) for name in present})
    with zipfile.ZipFile(out / 'evaluation_report.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in present + ['report_hashes.json']:
            archive.write(out / name, arcname=name)


def run(root, out, data, versions, hashes):
    import torch
    from transformers import GenerationConfig
    from peft import PeftModel
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.cuda.set_device(0)
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)
    tokenizer, prompts = validate_tokenizer(root, data)
    config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id)
    write_json(out / 'execution_plan.json', PLAN)
    write_json(out / 'prompt_audit.json', prompts)
    write_json(out / 'manifest.json', {
        'training_run': str(root), 'model': MODEL, 'revision': REVISION,
        'adapter_sha256': ADAPTER_SHA256, 'plan_sha256': PLAN_SHA256,
        'script_sha256': digest(__file__), 'versions': versions, 'training_artifact_sha256': hashes,
        'generation_config': config.to_dict(), 'gpu': torch.cuda.get_device_name(0),
        'schedule': [{'arm': arm, 'case_id': row['model_input']['case_id'],
                      'voice': row['model_input']['voice']} for arm in ARMS for row in prompts],
        'seed': 1111, 'training_performed': False, 'semantic_review': 'pending'})
    records = {arm: [] for arm in ARMS}
    for arm in ARMS:
        event(out, 'loading_model', arm=arm)
        model = load_base(torch)
        model.requires_grad_(False)
        # Check all model modules, not just the expected inventory: broad suffixes
        # must not silently capture excluded layers. This precedes base generation.
        verify_base_targets(torch, model, root, out, arm)
        if arm == 'saved_pilot_adapter_raw':
            model = PeftModel.from_pretrained(model, root / 'best_adapter', is_trainable=False)
            verify_loaded_adapter(model, root)
        model.eval()
        require(not any(p.requires_grad for p in model.parameters()), 'Evaluation parameters must be frozen')
        require({str(p.device) for p in model.parameters()} == {'cuda:0'}, 'Model is not fully on GPU 0')
        torch.cuda.reset_peak_memory_stats()
        for index, row in enumerate(prompts, 1):
            # Persist the attempt before invoking generation; failures never retry automatically.
            event(out, 'generation_start', arm=arm, index=index,
                  case_id=row['model_input']['case_id'], voice=row['model_input']['voice'])
            record = generate_one(torch, model, tokenizer, row, config)
            records[arm].append(record)
            write_json(out / f'{arm}.json', records[arm])
            event(out, 'response_saved', arm=arm, index=index,
                  mechanics_valid=record['mechanics']['valid'], ended_with_eos=record['ended_with_eos'])
        event(out, 'phase_finished', arm=arm, **memory(torch))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    require(digest(root / 'best_adapter/adapter_model.safetensors') == ADAPTER_SHA256,
            'Adapter file changed during evaluation')
    complete_outputs(out, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path, nargs='?')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--check-package', action='store_true',
                        help='Check portable files without ML packages or laptop artifacts')
    args = parser.parse_args()
    validate_package()
    if args.check_package:
        print('Package intact: eight states, 32 planned responses; no model loaded.')
        print('Grading agreement:', PLAN['review_protocol']['agreement'])
        return
    if not args.check_only:
        require(PLAN['review_protocol']['agreement'] == 'agreed',
                'Grading proposal is pending Travis; finalize it before fresh inference.')
    root = discover_run(args.run_directory)
    data, versions, hashes = validate_saved_run(root)
    print('Verified pilot:', root, flush=True)
    print('Adapter:', ADAPTER_SHA256, flush=True)
    if args.check_only:
        print('Artifact/config/package checks passed; tokenizer and GPU checks occur during execution.')
        return
    previous = sorted(root.glob('evaluation-002-*'))
    require(not previous, 'An evaluation directory already exists; inspect it before any repeat: '
            + ', '.join(str(p) for p in previous))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = root / f'evaluation-002-{stamp}-{uuid4().hex[:8]}'
    out.mkdir(exist_ok=False)
    print('Evaluation directory:', out, flush=True)
    write_json(out / 'execution_plan.json', PLAN)
    for name in ('evaluate_pilot_002.py', 'lora_smoke.py', 'evaluation-rubric-0.2.md'):
        shutil.copyfile(Path(__file__).with_name(name), out / name)
    try:
        run(root, out, data, versions, hashes)
    except BaseException as exc:
        (out / 'traceback.txt').write_text(traceback.format_exc())
        write_json(out / 'result.json', {'status': 'failed', 'error': str(exc),
                                       'training_performed': False, 'no_automatic_retry': True,
                                       'planned_generations': 32,
                                       'plan_sha256': PLAN_SHA256})
        package_report(out)
        raise
    package_report(out)
    print('Complete: 32 saved responses; human review pending. No training performed.', flush=True)
    print('Copy this report back:', out / 'evaluation_report.zip', flush=True)


if __name__ == '__main__':
    main()
