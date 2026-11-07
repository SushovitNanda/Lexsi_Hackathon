"""Export equal-size mixed-precision GGUFs with a matched random control.

Run --make-placeholder for the M8 mechanical test. It does not validate a circuit
or establish a safety benefit. Real D4 tables use --scores layer_x_matrix.json.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_data import sha256

MATRICES = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down']
EDITABLE = re.compile(r'^blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$')


def inventory(path):
    from gguf import GGUFReader
    reader = GGUFReader(str(path))
    tensors = {t.name: {'name': t.name, 'shape': [int(x) for x in t.shape],
        'n_elements': int(t.n_elements), 'bytes': int(t.n_bytes), 'type': t.tensor_type.name,
        'editable': bool(EDITABLE.fullmatch(t.name))} for t in reader.tensors}
    eos = reader.get_field('tokenizer.ggml.eos_token_id')
    template = reader.get_field('tokenizer.chat_template')
    return {'file': str(path.resolve()), 'sha256': sha256(path), 'bytes': path.stat().st_size,
            'eos_token_id': int(eos.parts[eos.data[0]][0]),
            'template_sha256': hashlib.sha256(template.parts[template.data[0]].tobytes()).hexdigest(),
            'data_offset': int(reader.data_offset), 'tensors': tensors,
            'by_type': dict(Counter(t['type'] for t in tensors.values()))}


def bytes_for(tensor, kind):
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
    block, size = GGML_QUANT_SIZES[GGMLQuantizationType[kind]]
    if tensor['shape'][0] % block or tensor['n_elements'] % block:
        raise ValueError(f'{tensor["name"]} is incompatible with {kind}')
    return tensor['n_elements'] // block * size


def predicted_bytes(stock, rules):
    # GGUF data are 32-byte aligned. Retain padding as well as raw tensor size.
    align = lambda n: (n + 31) // 32 * 32
    return stock['bytes'] + sum(align(bytes_for(stock['tensors'][name], kind)) -
        align(stock['tensors'][name]['bytes']) for name, kind in rules.items())


def load_scores(path, stock):
    data = json.loads(path.read_text())
    scores = {}
    for row in data['layer_x_matrix']:
        layer = int(row['layer'])
        for matrix in MATRICES:
            name = f'blk.{layer}.{matrix}.weight'
            value = float(row[matrix])
            if name in scores or not math.isfinite(value):
                raise ValueError(f'Duplicate or non-finite score: {name}')
            scores[name] = abs(value)
    expected = {name for name, t in stock['tensors'].items() if t['editable']}
    if scores.keys() != expected:
        raise ValueError('Scores must cover exactly the transformer layer matrices')
    return scores, data.get('kind', 'unvalidated_D4_table')


def solve(stock, scores, count=8, upgrade='Q8_0', downgrade='Q2_K', tolerance=.01):
    eligible = [n for n in scores if bytes_for(stock['tensors'][n], upgrade) > stock['tensors'][n]['bytes']]
    if not 0 < count <= len(eligible):
        raise ValueError('Invalid protected tensor count')
    protected = sorted(eligible, key=lambda n: (-scores[n], n))[:count]
    rules = {name: upgrade for name in protected}
    donors = sorted((n for n in scores if n not in rules and
        bytes_for(stock['tensors'][n], downgrade) < stock['tensors'][n]['bytes']), key=lambda n: (scores[n], n))
    for name in donors:
        if predicted_bytes(stock, rules) <= stock['bytes']:
            break
        rules[name] = downgrade
    predicted = predicted_bytes(stock, rules)
    if abs(predicted / stock['bytes'] - 1) > tolerance:
        raise ValueError('No feasible size within tolerance; reduce protected count or change types')
    return rules, protected


def random_control(stock, rules, seed=13):
    classes = defaultdict(list)
    for name, tensor in stock['tensors'].items():
        if tensor['editable']:
            classes[(tuple(tensor['shape']), tensor['type'])].append(name)
    actions = defaultdict(list)
    for name, kind in rules.items():
        tensor = stock['tensors'][name]
        actions[(tuple(tensor['shape']), tensor['type'])].append((name, kind))
    rng = random.Random(seed)
    control = {}
    for key in sorted(actions):
        selected = rng.sample(sorted(classes[key]), len(actions[key]))
        for (_, kind), name in zip(sorted(actions[key]), selected):
            control[name] = kind
    if control == rules:
        raise ValueError('Random control is identical; select another seed before any evaluation')
    if predicted_bytes(stock, control) != predicted_bytes(stock, rules):
        raise ValueError('Random control does not match precision/shape budget')
    return control


def write_rules(path, rules):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'^{re.escape(name)}$={kind.lower()}\n' for name, kind in sorted(rules.items())))


def export_rules(stock, changes):
    # Individual overrides can shift llama.cpp's heuristic K-mixture counters.
    # Force all matrix types to the measured stock layout, then apply changes.
    return {**{n: t['type'] for n, t in stock['tensors'].items() if len(t['shape']) > 1}, **changes}


def verify_build(path, stock, rules, tolerance):
    actual = inventory(path)
    if actual['tensors'].keys() != stock['tensors'].keys():
        raise ValueError('Output tensor names changed')
    if actual['eos_token_id'] != stock['eos_token_id'] or actual['template_sha256'] != stock['template_sha256']:
        raise ValueError('EOS or chat template metadata changed')
    for name, tensor in actual['tensors'].items():
        if tensor['shape'] != stock['tensors'][name]['shape'] or tensor['type'] != rules.get(name, stock['tensors'][name]['type']):
            raise ValueError(f'Unexpected output tensor shape/type: {name}')
    if actual['bytes'] != predicted_bytes(stock, rules):
        raise ValueError('Actual file size differs from block-size prediction')
    fraction = actual['bytes'] / stock['bytes'] - 1
    if abs(fraction) > tolerance:
        raise ValueError('Actual file size is outside the permitted budget')
    from gguf import GGUFReader
    reference = GGUFReader(stock['file'])
    output = GGUFReader(str(path))
    before = {t.name: t for t in reference.tensors}
    unchanged = 0
    for tensor in output.tensors:
        if tensor.name not in rules:
            if hashlib.sha256(memoryview(tensor.data)).digest() != hashlib.sha256(memoryview(before[tensor.name].data)).digest():
                raise ValueError(f'Unselected tensor changed: {tensor.name}')
            unchanged += 1
    return {'file': str(path.resolve()), 'sha256': actual['sha256'], 'bytes': actual['bytes'],
            'size_fraction_vs_stock': fraction, 'within_one_percent': abs(fraction) <= .01,
            'all_tensor_types_verified': True, 'unselected_tensors_byte_identical': unchanged,
            'by_type': actual['by_type']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT / 'models/gguf/orig-f16.gguf')
    ap.add_argument('--stock', type=Path, default=ROOT / 'models/gguf/orig-q3km.gguf')
    ap.add_argument('--scores', type=Path)
    ap.add_argument('--make-placeholder', action='store_true')
    ap.add_argument('--protected-count', type=int, default=8)
    ap.add_argument('--upgrade', default='Q8_0')
    ap.add_argument('--downgrade', default='Q2_K')
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--tolerance', type=float, default=.01)
    ap.add_argument('--out', type=Path, default=ROOT / 'models/gguf/placeholder')
    ap.add_argument('--report', type=Path, default=ROOT / 'results/mac/placeholder_protection_report.json')
    ap.add_argument('--artifacts-dir', type=Path, help='Separate inventory, score and rule files for this experiment')
    a = ap.parse_args()
    if a.make_placeholder == bool(a.scores):
        ap.error('Choose exactly one of --make-placeholder and --scores')
    if not 0 < a.tolerance <= .01:
        ap.error('Tolerance must be positive and no greater than 1%')
    stock = inventory(a.stock)
    if a.artifacts_dir:
        a.artifacts_dir.mkdir(parents=True, exist_ok=True)
    inv_path = a.artifacts_dir / 'stock_q3km_tensors.json' if a.artifacts_dir else ROOT / 'results/mac/stock_q3km_tensors.json'
    inv_path.write_text(json.dumps(stock, indent=2) + '\n')
    if a.make_placeholder:
        rng = random.Random(a.seed)
        a.scores = a.artifacts_dir / 'placeholder_scores.json' if a.artifacts_dir else ROOT / 'configs/mac/placeholder_scores.json'
        data = {'kind': 'placeholder_only_not_a_validated_circuit', 'seed': a.seed,
                'layer_x_matrix': [{'layer': layer, **{m: rng.random() for m in MATRICES}} for layer in range(36)]}
        a.scores.parent.mkdir(parents=True, exist_ok=True)
        a.scores.write_text(json.dumps(data, indent=2) + '\n')
    scores, kind = load_scores(a.scores, stock)
    rules, protected = solve(stock, scores, a.protected_count, a.upgrade, a.downgrade, a.tolerance)
    random_rules = random_control(stock, rules, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    llama = ROOT / 'KITS/llama.cpp'
    commit = subprocess.check_output(['git', '-C', str(llama), 'rev-parse', 'HEAD'], text=True).strip()
    spec = {'stock_sha256': stock['sha256'], 'source_f16_sha256': sha256(a.source),
        'llama_cpp_commit': commit, 'score_sha256': sha256(a.scores), 'score_kind': kind,
        'seed': a.seed, 'tolerance': a.tolerance, 'protected_count': a.protected_count,
        'baseline_type_policy': 'explicit measured stock type for every matrix',
        'protected_rules': rules, 'random_rules': random_rules, 'reported_RQ6_result': False}
    old = json.loads(a.report.read_text()) if a.report.exists() else None
    if old and any(old.get(k) != v for k, v in spec.items()):
        raise ValueError('An earlier export used different settings; preserve that report first')
    builds = {}
    for tag, selected in [('protected', rules), ('random', random_rules)]:
        rule_path = a.artifacts_dir / f'placeholder_{tag}.tensor-types.txt' if a.artifacts_dir else ROOT / f'configs/mac/placeholder_{tag}.tensor-types.txt'
        write_rules(rule_path, export_rules(stock, selected))
        target = a.out / f'{tag}-q3km.gguf'
        command = [str(llama / 'build/bin/llama-quantize'), '--tensor-type-file', str(rule_path), str(a.source)]
        if target.exists():
            if old is None or old.get('builds', {}).get(tag, {}).get('sha256') != sha256(target):
                raise ValueError('Existing build lacks matching verified provenance')
        else:
            dry = subprocess.run(command[:1] + ['--dry-run'] + command[1:] + ['Q3_K_M'],
                                 check=True, capture_output=True, text=True)
            (a.out / f'{tag}-dry-run.log').write_text(dry.stdout + dry.stderr)
            partial = target.with_suffix('.partial.gguf')
            with (a.out / f'{tag}-build.log').open('w') as log:
                subprocess.run(command + [str(partial), 'Q3_K_M'], check=True, stdout=log, stderr=subprocess.STDOUT)
            partial.replace(target)
        builds[tag] = verify_build(target, stock, selected, a.tolerance)
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps({**spec, 'stock_bytes': stock['bytes'], 'protected_tensors': protected,
            'predicted_bytes': predicted_bytes(stock, selected), 'builds': builds}, indent=2) + '\n')
        print(json.dumps({tag: builds[tag]}, indent=2), flush=True)
    if builds['protected']['bytes'] != builds['random']['bytes']:
        raise ValueError('Matched random and protected builds differ in size')
    report = json.loads(a.report.read_text())
    report['matched_random_size_exact'] = True
    report['random_overlap_with_protected_upgrades'] = len(set(protected) & {n for n, k in random_rules.items() if k == a.upgrade})
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print('Mechanical export test passed; no safety benefit is claimed.', flush=True)


if __name__ == '__main__':
    main()
