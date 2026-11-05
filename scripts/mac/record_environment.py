"""Record the Mac lane without overwriting Sushovit's environment table."""
import importlib.metadata as md
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    import torch
    def run(command):
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    llama = ROOT / 'KITS/llama.cpp'
    packages = ['torch', 'transformers', 'peft', 'accelerate', 'huggingface_hub', 'safetensors',
                'sentencepiece', 'auditkit', 'gguf', 'cmake', 'ninja']
    result = {'os': platform.platform(), 'architecture': platform.machine(),
        'python': platform.python_version(), 'chip': run(['sysctl', '-n', 'machdep.cpu.brand_string']),
        'memory_bytes': int(run(['sysctl', '-n', 'hw.memsize'])),
        'mps_available': torch.backends.mps.is_available(),
        'packages': {p: md.version(p) for p in packages},
        'llama_cpp_commit': run(['git', '-C', str(llama), 'rev-parse', 'HEAD']),
        'llama_server_version': run([str(llama / 'build/bin/llama-server'), '--version']),
        'devices': run([str(llama / 'build/bin/llama-cli'), '--list-devices']),
        'build': {'GGML_METAL': 'ON', 'GGML_METAL_EMBED_LIBRARY': 'ON',
                  'CMAKE_OSX_SYSROOT': '/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk'}}
    benchmark = ROOT / 'models/local-results/smoke-orig-q4km-dev200.summary.json'
    completed = json.loads(benchmark.read_text()) if benchmark.exists() else {}
    result['model_inference_verified'] = completed.get('completed') == 200 and not completed.get('errors')
    out = ROOT / 'results/mac/environment.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    marker = '\n## Manish — Mac lane (4 November 2026)\n'
    doc = (ROOT / 'ENV.md').read_text().split(marker)[0]
    lines = [marker, '\nIndependent GGUF lane; CUDA/NF4 stays on Sushovit’s machine. Full snapshot: `results/mac/environment.json`; exact Python dependency lock: `requirements-mac.lock`.\n',
        '| Item | Value |', '|---|---|', f'| Chip / RAM | {result["chip"]} / 24 GiB unified |',
        f'| Python | {result["python"]} (`.venv/`) |',
        f'| Metal | Built; Apple M5 detected; {"200-prompt inference verified" if result["model_inference_verified"] else "model inference pending"} |',
        f'| llama.cpp commit | `{result["llama_cpp_commit"]}` |',
        '| SDK | MacOSX26.5.sdk (explicitly selected from installed CLT SDKs) |']
    lines += [f'| {p} | {v} |' for p, v in result['packages'].items()]
    lines += ['', 'Build uses embedded Metal source and runtime compilation, so the standalone `metal` compiler is not required.',
              'This is a separate macOS PyTorch build; it does not change the reported T4 environment.', '']
    (ROOT / 'ENV.md').write_text(doc + '\n'.join(lines))
    print(f'Updated {out} and appended the Mac lane to ENV.md')


if __name__ == '__main__':
    main()
