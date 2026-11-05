"""Download pinned source snapshots; convert and quantize the original model.

Uses the user's local HF login (or HF_TOKEN); never logs credentials. Run hf auth
login and obtain gated-model access before running this script.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('HF_HOME', str(ROOT / 'models/hf_cache'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_data import sha256


def align_generation_eos(path, eos_token_id):
    """Preserve the model's generation EOS, which can differ from tokenizer EOS."""
    from gguf import GGUFReader
    reader = GGUFReader(str(path), 'r+')
    field = reader.get_field('tokenizer.ggml.eos_token_id')
    if field is None:
        raise ValueError('Converted GGUF has no EOS metadata')
    values = field.parts[field.data[0]]
    previous = int(values[0])
    values[0] = eos_token_id
    reader.data.flush()
    return previous


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download-only', action='store_true')
    a = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import GatedRepoError
    snapshots = {}
    out = ROOT / 'models/gguf'
    out.mkdir(parents=True, exist_ok=True)
    for short in ['global', 'base']:
        repo = f'CohereLabs/tiny-aya-{short}'
        info = HfApi().model_info(repo, files_metadata=True)
        try:
            path = snapshot_download(repo, revision=info.sha,
                allow_patterns=['*.json', '*.safetensors', '*.model', '*.jinja', '*.txt'])
        except GatedRepoError:
            if short == 'global':
                raise
            print('Base access is pending; continuing original Global conversion.', flush=True)
            continue
        source_hashes = {}
        published = {f.rfilename: f for f in info.siblings}
        for weights in sorted(Path(path).glob('*.safetensors')):
            actual = sha256(weights)
            expected = getattr(published[weights.name].lfs, 'sha256', None)
            # HF cache names may be Xet identifiers, not the raw file hash.
            if expected is None or actual != expected:
                raise ValueError(f'Downloaded shard checksum mismatch: {weights.name}')
            source_hashes[weights.name] = actual
        snapshots[short] = {'repo': repo, 'revision': info.sha, 'path': path, 'sha256': source_hashes}
    (out / 'source_snapshots.json').write_text(json.dumps(snapshots, indent=2) + '\n')
    if a.download_only:
        return
    llama = ROOT / 'KITS/llama.cpp'
    f16 = out / 'orig-f16.gguf'
    # Resume only when the recorded source and output hashes still match.
    provenance_path = out / 'orig-build.json'
    prior = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    commit = subprocess.check_output(['git', '-C', str(llama), 'rev-parse', 'HEAD'], text=True).strip()
    generation = json.loads((Path(snapshots['global']['path']) / 'generation_config.json').read_text())
    eos = generation['eos_token_id']
    if not isinstance(eos, int):
        raise ValueError('This recipe currently requires one generation EOS token')
    spec = {'sources': {'global': snapshots['global']}, 'llama_cpp_commit': commit,
            'metadata_overrides': {'tokenizer.ggml.eos_token_id': eos}}
    if prior and {k: prior[k] for k in spec} != spec:
        raise ValueError('Sources/tool version changed; preserve old outputs under another name first')
    outputs = prior.get('outputs', {})

    def build(path, command):
        if path.exists():
            if outputs.get(path.name, {}).get('sha256') != sha256(path):
                raise ValueError(f'Unverified existing output: {path}; move it aside before retrying')
            return
        partial = path.with_suffix('.partial.gguf')
        command = [str(partial) if str(x) == str(path) else str(x) for x in command]
        subprocess.run(command, check=True, cwd=ROOT)
        align_generation_eos(partial, eos)
        partial.replace(path)
        outputs[path.name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
        provenance_path.write_text(json.dumps({**spec, 'outputs': outputs}, indent=2) + '\n')

    build(f16, [sys.executable, llama / 'convert_hf_to_gguf.py', snapshots['global']['path'],
                '--outfile', f16, '--outtype', 'f16'])
    for quant in ['Q4_K_M', 'Q3_K_M']:
        target = out / f'orig-{quant.lower().replace("_", "")}.gguf'
        build(target, [llama / 'build/bin/llama-quantize', f16, target, quant])
    from audit_templates import audit
    audit(snapshots['global']['path'], f16, ROOT / 'results/mac/templates')


if __name__ == '__main__':
    main()
