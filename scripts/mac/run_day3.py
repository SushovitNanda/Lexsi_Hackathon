"""Run November 6 M9 cells sequentially; validate and package each finished cell.

No judge is substituted and no safety or fidelity gate is inferred from generation.
Run with the project venv. An exclusive lock prevents concurrent invocations.
"""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cell import load_rows, digest
from verify_data import sha256, verify

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / 'models/local-results/day3'
PUBLIC = ROOT / 'results/mac/nov6'
PROTOCOL = ROOT / 'MAC_PREREG_2026-11-06.md'
CELLS = [
    ('med_f16', 'med-f16.gguf', 'official', False),
    ('med_q4km', 'med-q4km.gguf', 'official', False),
    ('med_q3km', 'med-q3km.gguf', 'official', False),
    ('med_q4km-nopre', 'med-q4km.gguf', 'none', False),
    ('med_budget-phone', 'med-q3km.gguf', 'none', True),
]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def run(command, log):
    print('RUN', ' '.join(map(str, command)), flush=True)
    with log.open('a') as out:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True)


def validate_cell(cell, expected):
    path = LOCAL / f'{cell}.jsonl'
    records = [json.loads(s) for s in path.read_text().splitlines()]
    ids = [r['prompt_id'] for r in records]
    spec = json.loads(path.with_suffix('.run.json').read_text())
    summary = json.loads(path.with_suffix('.summary.json').read_text())
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f'{cell}: missing, duplicate or unexpected IDs')
    if not summary['complete'] or summary['errors'] or summary['completed'] != len(expected):
        raise ValueError(f'{cell}: incomplete summary')
    if {r['run_fingerprint'] for r in records} != {summary['run_fingerprint']}:
        raise ValueError(f'{cell}: mixed fingerprints')
    if digest({k: v for k, v in spec.items()}) != summary['run_fingerprint']:
        raise ValueError(f'{cell}: specification fingerprint mismatch')
    if spec['input_sha'] != json.loads((ROOT / 'configs/mac/nov5_inputs.json').read_text())['test_input_sha']:
        raise ValueError(f'{cell}: changed input digest')
    return summary, {'response_sha256': sha256(path), 'response_bytes': path.stat().st_size,
        'run_sha256': sha256(path.with_suffix('.run.json')), 'rows': len(ids),
        'by_set': dict(Counter(r['set'] for r in records)),
        'provisional': True, 'judged': False, 'fidelity_gate_passed': False}


def package(cell, expected):
    summary, evidence = validate_cell(cell, expected)
    write_json(PUBLIC / f'{cell}.summary.json', summary)
    write_json(PUBLIC / f'{cell}.validation.json', evidence)
    archive = LOCAL / f'D6_Manish_2026-11-06_{cell}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for suffix in ['.jsonl', '.run.json', '.summary.json']:
            path = LOCAL / f'{cell}{suffix}'
            z.write(path, f'cells/{path.name}')
        for path in [PROTOCOL, ROOT / 'MAC_PREREG_2026-11-05.md',
                     ROOT / 'configs/mac/nov5_inputs.json',
                     ROOT / 'references/SUSHOVIT_PREREG_2026-11-05.md',
                     PUBLIC / f'{cell}.validation.json']:
            z.write(path, str(path.relative_to(ROOT)))
        z.writestr('README.txt', f'''Manish November 6 handoff: {cell}, {len(expected)} responses.
Generation complete, provisional pending current merge parity and >=48/50 shared judged fidelity.
Use centralized GPU-lane judging. Adopt November 5 PREREG Amendment 1:
EN/Hinglish Qwen3Guard safety; Hindi WildGuard on IndicTrans2 back-translation.
ORR uses Qwen3Guard in all languages; Hinglish ORR directional only.
Controversial safe primary / unsafe sensitivity for Qwen3Guard.
Join prompt_id; response contains the generated answer. Do not infer labels from scripts or token counts.
EOS3 preserved uniformly; known medical repeated hidden EOS6/256-token-cap risk disclosed in Mac protocol.
No ASR/ORR/utility or cross-platform fidelity pass claimed. MedMCQA not included.
Repository upload is not confirmation of collaborator receipt.
''')
    return {**evidence, 'archive': str(archive.relative_to(ROOT)), 'archive_sha256': sha256(archive),
            'archive_bytes': archive.stat().st_size, 'summary': summary}


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    with (LOCAL / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'started_at': stamp(), 'pid': os.getpid(), 'status': 'validating', 'completed_cells': {}}
        def save():
            state['updated_at'] = stamp()
            write_json(LOCAL / 'SESSION_STATE.json', state)
            write_json(PUBLIC / 'status.json', state)
        save()
        try:
            if not verify(ROOT / 'Datasets/final')['evaluation_verified']:
                raise ValueError('Evaluation checksum verification failed')
            frozen = json.loads((ROOT / 'configs/mac/nov5_inputs.json').read_text())
            rows = load_rows(sorted((ROOT / 'Datasets/final').glob('*/prompts.jsonl')), 'test')
            if digest(rows) != frozen['test_input_sha'] or [r['id'] for r in rows] != frozen['test_prompt_ids']:
                raise ValueError('November 5 input population/order changed')
            expected = set(frozen['test_prompt_ids'])
            source = ROOT / 'models/gguf/med-f16.gguf'
            manifest = json.loads((ROOT / 'results/mac/medical_gguf_build.json').read_text())
            if sha256(source) != manifest['sha256']:
                raise ValueError('Medical F16 provenance changed')
            for suffix, kind in [('q4km', 'Q4_K_M'), ('q3km', 'Q3_K_M')]:
                dest = ROOT / f'models/gguf/med-{suffix}.gguf'
                receipt = PUBLIC / f'med-{suffix}.build.json'
                state.update(status='quantizing', active_cell=kind); save()
                if dest.exists():
                    if not receipt.exists() or sha256(dest) != json.loads(receipt.read_text())['sha256']:
                        raise ValueError(f'Unverified existing quantization: {dest}')
                else:
                    partial = dest.with_suffix('.partial.gguf')
                    run([ROOT / 'KITS/llama.cpp/build/bin/llama-quantize', source, partial, kind, '8'],
                        LOCAL / f'quantize-{suffix}.log')
                    from gguf import GGUFReader
                    from audit_templates import gguf_string
                    a, b = GGUFReader(str(source)), GGUFReader(str(partial))
                    field = b.get_field('tokenizer.ggml.eos_token_id')
                    if int(field.parts[field.data[0]][0]) != 3 or gguf_string(a, 'tokenizer.chat_template') != gguf_string(b, 'tokenizer.chat_template'):
                        raise ValueError('Quantizer changed EOS/template')
                    if len(a.tensors) != len(b.tensors) or {t.name: tuple(t.shape) for t in a.tensors} != {t.name: tuple(t.shape) for t in b.tensors}:
                        raise ValueError('Quantizer changed tensor population')
                    del a, b
                    partial.rename(dest)
                    write_json(receipt, {'source_sha256': manifest['sha256'], 'quantization': kind,
                        'sha256': sha256(dest), 'bytes': dest.stat().st_size, 'created_at': stamp(),
                        'llama_cpp_commit': manifest['llama_cpp_commit'], 'eos_token_id': 3,
                        'cross_machine_merge_parity_verified': False})
            for cell, model, template, phone in CELLS:
                state.update(status='generating', active_cell=cell); save()
                summary_path = LOCAL / f'{cell}.summary.json'
                if summary_path.exists() and json.loads(summary_path.read_text()).get('complete'):
                    state['completed_cells'][cell] = package(cell, expected); save(); continue
                log = LOCAL / f'{cell}.server.log'
                receipt = log.with_suffix('.receipt.json')
                receipt.unlink(missing_ok=True)
                serve_cmd = [sys.executable, ROOT / 'scripts/mac/serve.py', '--gguf', ROOT / 'models/gguf' / model,
                             '--template', template, '--log', log]
                if phone:
                    serve_cmd.append('--phone-cache')
                with (LOCAL / f'{cell}.launcher.log').open('a') as launchlog:
                    server = subprocess.Popen(list(map(str, serve_cmd)), cwd=ROOT, stdout=launchlog, stderr=subprocess.STDOUT)
                    try:
                        for _ in range(300):
                            if server.poll() is not None:
                                raise RuntimeError(f'{cell}: server exited during startup; inspect launcher log')
                            if receipt.exists():
                                break
                            time.sleep(1)
                        else:
                            raise TimeoutError('Server receipt timeout')
                        run([sys.executable, ROOT / 'scripts/mac/run_cell.py', '--cell', cell,
                             '--gguf', ROOT / 'models/gguf' / model, '--tokenizer', ROOT / 'models/merged/med',
                             '--server-receipt', receipt, '--protocol', PROTOCOL, '--split', 'test',
                             '--template', template, '--cache-k', 'q4_0' if phone else 'f16',
                             '--cache-v', 'q4_0' if phone else 'f16', '--out', LOCAL / f'{cell}.jsonl'],
                            LOCAL / f'{cell}.generation.log')
                    finally:
                        if receipt.exists():
                            native = json.loads(receipt.read_text())
                            live = subprocess.run(['ps', '-p', str(native['pid']), '-o', 'command='], capture_output=True, text=True).stdout.strip()
                            if live == ' '.join(native['command']):
                                os.kill(native['pid'], signal.SIGTERM)
                        try:
                            server.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            server.terminate()
                            server.wait(timeout=15)
                state['completed_cells'][cell] = package(cell, expected); save()
                print(f'COMPLETE {cell}', flush=True)
            state.update(status='complete', active_cell=None, finished_at=stamp(),
                         completed_responses=sum(c['rows'] for c in state['completed_cells'].values()),
                         external_blockers=['current reference merge parity', 'shared judged fidelity50 gate', 'centralized response judgments'])
            save()
        except BaseException as error:
            state.update(status='failed', error=repr(error)); save()
            raise


if __name__ == '__main__':
    main()
