"""Standalone D6 helper for Sushovit's T4: generate the exact shared 50 prompts.

This reference helper runs on Colab/CUDA, not on the Mac. It does not judge
responses or assert the 48/50 fidelity gate. Copy it with the private input file
and configs/mac/nov5_inputs.json. Dependencies match the pinned team HF stack.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import unicodedata


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def script_of(text):
    scripts = set()
    for c in text:
        if unicodedata.category(c).startswith('L'):
            name = unicodedata.name(c, '')
            scripts.add('devanagari' if 'DEVANAGARI' in name else 'latin' if 'LATIN' in name else 'other')
    return '+'.join(sorted(scripts)) or 'none'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True, help='Current medical merged FP16 directory')
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0).upper():
        raise ValueError('The registered cross-engine reference requires the Colab T4')
    rows = [json.loads(line) for line in a.input.read_text(encoding='utf-8').splitlines()]
    expected = json.loads(a.manifest.read_text(encoding='utf-8'))['fidelity']
    if len(rows) != 50 or digest(rows) != expected['input_sha'] or [r['id'] for r in rows] != expected['prompt_ids']:
        raise ValueError('Inputs do not match the committed 50-prompt manifest')
    info = json.loads((a.model / 'merge_info.json').read_text(encoding='utf-8'))
    hashes = {}
    for p in sorted(a.model.glob('*.safetensors')):
        hashes[p.name] = file_hash(p)
        with safe_open(p, framework='pt', device='cpu') as f:
            if any(f.get_slice(k).get_dtype() != 'F16' for k in f.keys()):
                raise ValueError('Re-save the current reference model as FP16 before this run')
    if not hashes or hashes != info['sha256']:
        raise ValueError('Reference weights differ from their merge manifest')
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map='cuda', local_files_only=True)
    model.eval()
    if model.generation_config.eos_token_id != 3:
        raise ValueError('Source generation EOS differs from the registered ID 3')
    template = tok.get_chat_template()
    spec = {'cell_id': 'med_fp16_fidelity50', 'purpose': 'T4_fidelity_reference_not_a_reported_test_cell',
        'model_sha256': hashes, 'input_sha': expected['input_sha'], 'template_sha': hashlib.sha256(template.encode()).hexdigest(),
        'temperature': 0, 'seed': 13, 'max_tokens': 256, 'eos_token_id': 3,
        'gpu': torch.cuda.get_device_name(0), 'torch': md.version('torch'), 'transformers': md.version('transformers')}
    fingerprint = digest(spec)
    completed = {}
    if a.out.exists():
        data = a.out.read_bytes()
        if data and not data.endswith(b'\n'):
            raise ValueError('Incomplete trailing JSONL record; preserve it and restore the last full line before resuming')
        for line in data.splitlines():
            record = json.loads(line)
            if record['run_fingerprint'] != fingerprint or record['prompt_id'] in completed:
                raise ValueError('Resume identity or duplicate-ID failure')
            completed[record['prompt_id']] = record
    if completed.keys() - set(expected['prompt_ids']):
        raise ValueError('Output contains unexpected prompt IDs')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix('.run.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    with a.out.open('a', encoding='utf-8') as output:
        for row in rows:
            if row['id'] in completed:
                continue
            rendered = tok.apply_chat_template([{'role': 'user', 'content': row['text']}], tokenize=False, add_generation_prompt=True)
            encoded = tok(rendered, add_special_tokens=False, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                tokens = model.generate(**encoded, do_sample=False, max_new_tokens=256)
            response = tok.decode(tokens[0, encoded.input_ids.shape[1]:], skip_special_tokens=True)
            record = {**row, **spec, 'prompt_id': row['id'], 'response': response,
                'answer_script': script_of(response), 'run_fingerprint': fingerprint,
                'completion_tokens': int(tokens.shape[1] - encoded.input_ids.shape[1]),
                'terminal_token_id': int(tokens[0, -1].item()),
                'end_turn_6_count': int((tokens[0, encoded.input_ids.shape[1]:] == 6).sum().item()),
                'generated_at': datetime.now(timezone.utc).isoformat()}
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            os.fsync(output.fileno())
            completed[row['id']] = record
            print(f'{len(completed)}/50', flush=True)
    print('T4 reference complete. Judge paired responses on the same pinned judge; no gate is claimed here.')


if __name__ == '__main__':
    main()
