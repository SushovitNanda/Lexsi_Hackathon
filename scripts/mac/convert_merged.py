"""Convert an already-validated local FP16 merge into a reproducible F16 GGUF."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_data import sha256
from prepare_models import align_generation_eos
from audit_templates import audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    from safetensors import safe_open
    files = sorted(a.model.glob('*.safetensors'))
    if not files:
        raise ValueError('No source model shards')
    hashes = {}
    for path in files:
        hashes[path.name] = sha256(path)
        with safe_open(path, framework='pt', device='cpu') as f:
            for name in f.keys():
                if f.get_slice(name).get_dtype() != 'F16':
                    raise ValueError(f'Source merge is not FP16: {name}')
    info = json.loads((a.model / 'merge_info.json').read_text())
    if info['sha256'] != hashes:
        raise ValueError('Source merge differs from its local manifest')
    eos = json.loads((a.model / 'generation_config.json').read_text())['eos_token_id']
    if not isinstance(eos, int):
        raise ValueError('One EOS token is required')
    llama = ROOT / 'KITS/llama.cpp'
    spec = {'source_model': str(a.model.resolve()), 'source_sha256': hashes,
        'base_revision': info.get('base_revision'),
        'llama_cpp_commit': subprocess.check_output(['git', '-C', str(llama), 'rev-parse', 'HEAD'], text=True).strip(),
        'metadata_overrides': {'tokenizer.ggml.eos_token_id': eos},
        'cross_machine_merge_parity_verified': False}
    manifest = a.out.with_suffix('.build.json')
    if a.out.exists():
        previous = json.loads(manifest.read_text())
        if any(previous.get(k) != v for k, v in spec.items()) or previous['sha256'] != sha256(a.out):
            raise ValueError('Existing output differs; preserve it under a different name')
    else:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        partial = a.out.with_suffix('.partial.gguf')
        subprocess.run([sys.executable, str(llama / 'convert_hf_to_gguf.py'), str(a.model),
                        '--outfile', str(partial), '--outtype', 'f16'], check=True)
        align_generation_eos(partial, eos)
        partial.replace(a.out)
        manifest.write_text(json.dumps({**spec, 'sha256': sha256(a.out), 'bytes': a.out.stat().st_size}, indent=2) + '\n')
    report = ROOT / f'results/mac/{a.tag}_gguf_build.json'
    report.write_bytes(manifest.read_bytes())
    audit(str(a.model), a.out, ROOT / f'results/mac/{a.tag}_templates')
    print(f'Converted and audited: {a.out}', flush=True)


if __name__ == '__main__':
    main()
