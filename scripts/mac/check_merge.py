"""Validate FP16 merged tensors and compare D2 hashes (optional reference weights)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_data import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--expected', type=Path, help='D2 merge_info.json from Sushovit; omit for local validation only')
    ap.add_argument('--reference', type=Path, help='Optional reference safetensors directory, to measure max abs difference')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    from safetensors import safe_open
    expected = json.loads(a.expected.read_text())['sha256'] if a.expected else None
    files = sorted(a.model.glob('*.safetensors'))
    if not files:
        raise ValueError('No merged safetensors found')
    hashes, dtypes, tensors = {}, {}, {}
    import torch
    for file in files:
        hashes[file.name] = sha256(file)
        with safe_open(file, framework='pt', device='cpu') as f:
            for name in f.keys():
                dtype = f.get_slice(name).get_dtype()
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
                if (dtype.startswith('F') or dtype.startswith('BF')) and dtype != 'F16':
                    raise ValueError(f'{file.name}/{name} has {dtype}; FP16 required')
                if not torch.isfinite(f.get_tensor(name)).all():
                    raise ValueError(f'{file.name}/{name} has non-finite values')
                tensors[name] = file
    result = {'hashes': hashes, 'expected_hashes': expected,
              'hashes_match': hashes == expected if expected is not None else None,
              'reference_supplied': expected is not None, 'all_tensors_finite': True, 'dtypes': dtypes}
    if a.reference:
        reference = {}
        for file in a.reference.glob('*.safetensors'):
            with safe_open(file, framework='pt', device='cpu') as f:
                reference.update({k: file for k in f.keys()})
        if tensors.keys() != reference.keys():
            raise ValueError('Reference tensor names differ')
        maximum = 0.0
        import torch
        for name, file in tensors.items():
            with safe_open(file, framework='pt', device='cpu') as f, safe_open(reference[name], framework='pt', device='cpu') as g:
                x, y = f.get_tensor(name), g.get_tensor(name)
                if x.shape != y.shape:
                    raise ValueError(f'Shape mismatch: {name}')
                if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise ValueError(f'Non-finite tensor: {name}')
                maximum = max(maximum, (x.float() - y.float()).abs().max().item())
        result['max_absolute_difference'] = maximum
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if expected is not None and not result['hashes_match']:
        if not a.reference:
            print('Hashes differ; reference tensors are required to measure max absolute difference.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
