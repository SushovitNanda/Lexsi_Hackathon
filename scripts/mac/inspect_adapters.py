"""Inspect D2 adapters without loading the gated base model or changing files."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_data import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, default=Path('runs'))
    ap.add_argument('--out', type=Path, default=Path('results/mac/adapter_inventory.json'))
    a = ap.parse_args()
    from safetensors import safe_open
    import torch
    result = {}
    for tag in ['med', 'ctrl', 'med_e2']:
        adapter = a.runs / tag / 'adapter'
        config = json.loads((adapter / 'adapter_config.json').read_text())
        if config['base_model_name_or_path'] != 'CohereLabs/tiny-aya-global':
            raise ValueError(f'{tag}: incorrect base model')
        tensors, finite = {}, True
        weights = adapter / 'adapter_model.safetensors'
        with safe_open(weights, framework='pt', device='cpu') as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                finite = finite and bool(torch.isfinite(tensor).all())
                tensors[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
        if not finite:
            raise ValueError(f'{tag}: non-finite adapter weights')
        result[tag] = {'adapter_path': str(adapter.resolve()), 'base_model': config['base_model_name_or_path'],
            'lora': {k: config[k] for k in ['r', 'lora_alpha', 'lora_dropout', 'target_modules']},
            'sha256': {p.name: sha256(p) for p in sorted(adapter.iterdir()) if p.is_file()},
            'tensor_count': len(tensors), 'tensors': tensors, 'finite': finite,
            'train_summary': json.loads((a.runs / tag / 'train_summary.json').read_text())}
        result[tag]['lora']['target_modules'] = sorted(result[tag]['lora']['target_modules'])
    if result['med']['lora'] != result['ctrl']['lora']:
        raise ValueError('Medical and control LoRA configuration differs')
    if result['med']['sha256']['chat_template.jinja'] != result['ctrl']['sha256']['chat_template.jinja']:
        raise ValueError('Medical/control chat templates differ')
    if result['med']['tensors'] != result['ctrl']['tensors']:
        raise ValueError('Medical/control adapter tensor names, shapes or dtypes differ')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({tag: {'tensor_count': r['tensor_count'], 'finite': r['finite']} for tag, r in result.items()}, indent=2))


if __name__ == '__main__':
    main()
