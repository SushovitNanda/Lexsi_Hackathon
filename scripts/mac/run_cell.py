"""Resumable GGUF generation through AuditKIT's api backend.

Templates are rendered locally and sent to /v1/completions to prevent the server
from silently substituting a template. Response files are private hand-offs.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_data import sha256

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def answer_script(text):
    scripts = set()
    for ch in text:
        if unicodedata.category(ch).startswith('L'):
            name = unicodedata.name(ch, '')
            scripts.add('devanagari' if 'DEVANAGARI' in name else 'latin' if 'LATIN' in name else 'other')
    return '+'.join(sorted(scripts)) or 'none'


def load_rows(paths, split):
    rows, ids = [], set()
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['split'] != split or row['set'] == 'medmcqa':
                continue
            if row['id'] in ids:
                raise ValueError(f'Duplicate prompt id: {row["id"]}')
            ids.add(row['id'])
            rows.append(row)
    if not rows:
        raise ValueError('No prompts selected')
    return rows


def resume_rows(path, fingerprint):
    if not path.exists():
        return {}
    data = path.read_bytes()
    # A crash may tear the last write. Preserve the partial tail for inspection.
    if data and not data.endswith(b'\n'):
        last = data.rfind(b'\n') + 1
        path.with_suffix('.partial').write_bytes(data[last:])
        data = data[:last]
        path.write_bytes(data)
    completed = {}
    for line in data.splitlines():
        row = json.loads(line)
        if row['run_fingerprint'] != fingerprint:
            raise ValueError('Resume refused: model, inputs, template or settings changed')
        if row['prompt_id'] in completed:
            raise ValueError('Duplicate response in resume file')
        completed[row['prompt_id']] = row
    return completed


def one_prompt(row, rendered, args, provenance, input_ids=None):
    import auditkit as ak
    from auditkit.model.api_gen import APIModel
    start = time.monotonic()
    # An instance per worker avoids sharing AuditKIT's non-threadsafe Session.
    # llama.cpp automatically adds BOS to string completion prompts. Explicit
    # token IDs preserve the template's existing BOS exactly once.
    extra = {'prompt': input_ids} if input_ids is not None else {}
    backend = APIModel(model=args.model_id, api_base=args.api_base,
                       api_key='local-no-key', chat_template=False, cache_prompt=False, **extra)
    result = ak.evaluate([ak.Sample(input=rendered, id=row['id'], metadata=provenance)],
        model=backend, scorers=[], config=ak.RunConfig(temperature=0, seed=13,
        max_tokens=args.max_tokens, batch_size=1, timeout=args.timeout, max_retries=2,
        extra=provenance))
    prediction = result.predictions[0]
    if result.errors or prediction.raw_output is None or prediction.metadata.get('error'):
        raise RuntimeError(f'Generation failed for {row["id"]}: {result.errors}')
    response = prediction.raw_output
    return {**row, 'prompt_id': row['id'], 'response': response,
            'answer_script': answer_script(response), **provenance,
            'auditkit_fingerprint': result.fingerprint,
            'prompt_sha': hashlib.sha256(rendered.encode()).hexdigest(),
            'seconds': time.monotonic() - start,
            'token_usage': result.token_usage,
            'generated_at': datetime.now(timezone.utc).isoformat()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--gguf', type=Path, required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--template', choices=['official', 'none'], default='official')
    ap.add_argument('--api-base', default='http://127.0.0.1:8080/v1')
    ap.add_argument('--model-id', default='tiny-aya')
    ap.add_argument('--server-receipt', type=Path, required=True,
                    help='Receipt written by scripts/mac/serve.py')
    ap.add_argument('--inputs', type=Path, nargs='+')
    ap.add_argument('--split', choices=['dev', 'test', 'calib'], default='dev')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--timeout', type=float, default=600)
    ap.add_argument('--cache-k', default='f16')
    ap.add_argument('--cache-v', default='f16')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, help='Committed Mac pre-registration; required for test cells')
    args = ap.parse_args()
    if args.workers < 1 or args.max_tokens < 1 or (args.limit is not None and args.limit < 1):
        ap.error('workers, max-tokens and limit must be positive')
    from urllib.parse import urlsplit
    if urlsplit(args.api_base).hostname not in {'127.0.0.1', 'localhost', '::1'}:
        ap.error('This harness only sends evaluation prompts to a local llama-server')
    import requests
    import auditkit as ak
    from transformers import AutoTokenizer
    from gguf import GGUFReader
    from audit_templates import gguf_string

    receipt = json.loads(args.server_receipt.read_text())
    import subprocess
    protocol = {}
    if args.split == 'test' and args.protocol is None:
        ap.error('Test generation requires --protocol pointing to a committed pre-registration')
    if args.protocol:
        relative = args.protocol.resolve().relative_to(ROOT)
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'log', '-1', '--format=%H', '--', str(relative)], text=True).strip()
        if not commit:
            raise ValueError('Protocol has not been committed')
        frozen = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{relative}'])
        if frozen != args.protocol.read_bytes():
            raise ValueError('Protocol differs from its committed version')
        protocol = {'prereg_commit': commit, 'prereg_sha256': sha256(args.protocol)}
    live_command = subprocess.check_output(['ps', '-p', str(receipt['pid']), '-o', 'command='], text=True).strip()
    if live_command != ' '.join(receipt['command']):
        raise ValueError('Server receipt does not match the running process')
    if receipt['cache_k'] != args.cache_k or receipt['cache_v'] != args.cache_v:
        raise ValueError('Requested cache types differ from live server launch receipt')

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    reader = GGUFReader(str(args.gguf))
    stored = gguf_string(reader, 'tokenizer.chat_template')
    eos_field = reader.get_field('tokenizer.ggml.eos_token_id')
    eos = int(eos_field.parts[eos_field.data[0]][0])
    source_eos = json.loads((Path(args.tokenizer) / 'generation_config.json').read_text())['eos_token_id']
    if eos != source_eos:
        raise ValueError('GGUF EOS differs from the source generation config; rebuild with prepare_models.py')
    if stored != tok.get_chat_template():
        raise ValueError('GGUF/HF template mismatch: run audit_templates.py first')
    template = stored if args.template == 'official' else (ROOT / 'templates/none.jinja').read_text()
    if args.template == 'official' and 'contextual safety mode' not in template.lower():
        raise ValueError('Official safety preamble not found')
    paths = args.inputs or sorted((ROOT / 'Datasets/final').glob('*/prompts.jsonl'))
    rows = load_rows(paths, args.split)
    if args.limit:
        rows = rows[:args.limit]
    # Verify that the live server loaded this exact local file. The runbook uses
    # an absolute -m path, and records effective runtime settings via /props.
    base = args.api_base.removesuffix('/v1')
    props_response = requests.get(base + '/props', timeout=30)
    props_response.raise_for_status()
    props = props_response.json()
    models_response = requests.get(args.api_base + '/models', timeout=30)
    models_response.raise_for_status()
    models = models_response.json()['data']
    advertised = [m for m in models if m['id'] == args.model_id]
    if not advertised or Path(props.get('model_path', '')).resolve() != args.gguf.resolve():
        raise ValueError('Live server model alias/path differs from --model-id/--gguf')
    source_commit = __import__('subprocess').check_output(
        ['git', '-C', str(ROOT / 'KITS/llama.cpp'), 'rev-parse', 'HEAD'], text=True).strip()
    model_sha = sha256(args.gguf)
    if model_sha != receipt['model_sha256']:
        raise ValueError('Server receipt model hash differs from --gguf')
    spec = {'cell_id': args.cell, 'model_sha256': model_sha,
            'template_sha': hashlib.sha256(template.encode()).hexdigest(),
            'template_mode': args.template, 'max_tokens': args.max_tokens, 'temperature': 0,
            'seed': 13, 'cache_k': args.cache_k, 'cache_v': args.cache_v,
            'cache_prompt': False, 'prompt_format': 'token_ids',
            'eos_token_id': eos,
            'llama_cpp_commit': source_commit, 'auditkit_version': ak.__version__,
            'input_sha': digest(rows), 'server_props': props, 'workers': args.workers}
    spec.update(protocol)
    spec['implementation_sha256'] = {name: sha256(ROOT / name) for name in
        ['scripts/mac/run_cell.py', 'scripts/mac/serve.py', 'templates/none.jinja', 'requirements-mac.lock']}
    fingerprint = digest(spec)
    provenance = {k: v for k, v in spec.items() if k != 'server_props'}
    provenance['run_fingerprint'] = fingerprint
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = resume_rows(args.out, fingerprint)
        if completed.keys() - {r['id'] for r in rows}:
            raise ValueError('Resume file contains unselected prompt ids')
        args.out.with_suffix('.run.json').write_text(json.dumps(spec, indent=2) + '\n')
        pending = [r for r in rows if r['id'] not in completed]
        start = time.monotonic()
        errors = []
        prepared = []
        for row in pending:
                rendered = tok.apply_chat_template([{'role': 'user', 'content': row['text']}],
                    chat_template=template, tokenize=False, add_generation_prompt=True)
                input_ids = tok(rendered, add_special_tokens=False).input_ids
                slot_context = props['default_generation_settings']['n_ctx']
                if len(input_ids) + args.max_tokens > slot_context:
                    raise ValueError(f'Prompt plus output budget exceeds slot context: {row["id"]}')
                tokenized = requests.post(base + '/tokenize', json={'content': rendered,
                    'add_special': False, 'parse_special': True}, timeout=30)
                tokenized.raise_for_status()
                if tokenized.json()['tokens'] != input_ids:
                    raise ValueError(f'HF/GGUF token mismatch: {row["id"]}')
                prepared.append((row, rendered, input_ids))
        with args.out.open('a', encoding='utf-8') as output, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one_prompt, row, rendered, args, provenance, input_ids): row['id']
                       for row, rendered, input_ids in prepared}
            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception as e:
                    errors.append({'prompt_id': futures[future], 'error': str(e)})
                    continue
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
                output.flush()
                __import__('os').fsync(output.fileno())
                completed[record['prompt_id']] = record
                print(f'{len(completed)}/{len(rows)}', flush=True)
        elapsed = time.monotonic() - start
        new_ids = {r['id'] for r in pending}
        new_tokens = sum((r.get('token_usage') or {}).get('completion_tokens', 0)
                         for k, r in completed.items() if k in new_ids)
        summary = {'cell_id': args.cell, 'run_fingerprint': fingerprint,
            'selected': len(rows), 'completed': len(completed), 'new_responses': len(pending) - len(errors),
            'session_seconds': elapsed, 'session_responses_per_second': (len(pending)-len(errors))/max(elapsed, 1e-9),
            'session_completion_tokens': new_tokens, 'session_tokens_per_second': new_tokens/max(elapsed, 1e-9),
            'by_lang': dict(Counter(r['lang'] for r in completed.values())), 'errors': errors}
        summary['by_set'] = dict(Counter(r['set'] for r in completed.values()))
        summary['complete'] = len(completed) == len(rows) and not errors
        args.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
