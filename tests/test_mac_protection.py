from collections import Counter
import math
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/mac'))
from protect_gguf import MATRICES, bytes_for, export_rules, predicted_bytes, random_control, solve


def fixture_stock():
    tensors = {}
    for layer in range(36):
        for matrix in MATRICES:
            name = f'blk.{layer}.{matrix}.weight'
            shape = [11008, 2048] if matrix == 'ffn_down' else [2048, 11008] if matrix.startswith('ffn') else [2048, 512] if matrix in ['attn_k', 'attn_v'] else [2048, 2048]
            kind = 'Q3_K' if matrix in ['attn_q', 'ffn_gate', 'ffn_up'] else 'Q4_K'
            tensor = {'name': name, 'shape': shape, 'n_elements': math.prod(shape), 'type': kind, 'editable': True}
            tensor['bytes'] = bytes_for(tensor, kind)
            tensors[name] = tensor
    name = 'token_embd.weight'
    tensor = {'name': name, 'shape': [2048, 262144], 'n_elements': 2048 * 262144, 'type': 'Q6_K', 'editable': False}
    tensor['bytes'] = bytes_for(tensor, 'Q6_K')
    tensors[name] = tensor
    tensors['output_norm.weight'] = {'name': 'output_norm.weight', 'shape': [2048], 'n_elements': 2048, 'type': 'F32', 'bytes': 8192, 'editable': False}
    return {'bytes': 4096 + sum(t['bytes'] for t in tensors.values()), 'tensors': tensors}


def test_solver_and_random_control_have_exact_matched_precision_budget():
    stock = fixture_stock()
    rng = random.Random(13)
    scores = {name: rng.random() for name, t in stock['tensors'].items() if t['editable']}
    rules, protected = solve(stock, scores)
    control = random_control(stock, rules)
    assert len(protected) == 8
    assert protected == sorted(scores, key=lambda n: (-scores[n], n))[:8]
    assert all(stock['tensors'][n]['editable'] for n in rules.keys() | control.keys())
    assert abs(predicted_bytes(stock, rules) / stock['bytes'] - 1) <= .01
    assert predicted_bytes(stock, rules) == predicted_bytes(stock, control)
    actions = lambda selected: Counter((tuple(stock['tensors'][n]['shape']), stock['tensors'][n]['type'], kind) for n, kind in selected.items())
    assert actions(rules) == actions(control)
    assert rules != control


def test_export_pins_every_matrix_to_prevent_k_mixture_drift():
    stock = fixture_stock()
    changes = {'blk.0.attn_q.weight': 'Q8_0'}
    full = export_rules(stock, changes)
    assert full['blk.0.attn_q.weight'] == 'Q8_0'
    assert full['blk.3.attn_v.weight'] == stock['tensors']['blk.3.attn_v.weight']['type']
    assert full['token_embd.weight'] == 'Q6_K'
    assert 'output_norm.weight' not in full
    assert len(full) == 253


def test_infeasible_budget_is_rejected():
    stock = fixture_stock()
    scores = {name: 1.0 for name, t in stock['tensors'].items() if t['editable']}
    with pytest.raises(ValueError, match='No feasible size'):
        solve(stock, scores, count=252)
