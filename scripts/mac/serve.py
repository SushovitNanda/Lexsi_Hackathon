"""Start the pinned Metal llama-server with the plan's 8 slots/16384 context."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_data import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gguf', type=Path, required=True)
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--phone-cache', action='store_true')
    ap.add_argument('--template', choices=['official', 'none'], default='official')
    ap.add_argument('--log', type=Path, default=ROOT / 'models/local-results/server.log')
    a = ap.parse_args()
    a.log.parent.mkdir(parents=True, exist_ok=True)
    binary = ROOT / 'KITS/llama.cpp/build/bin/llama-server'
    command = [str(binary), '-m', str(a.gguf.resolve()), '--alias', 'tiny-aya',
               '--host', '127.0.0.1', '--port', str(a.port), '-ngl', '99',
               '-np', '8', '-c', '16384', '-fa', 'on', '--jinja', '-ctk',
               'q4_0' if a.phone_cache else 'f16', '-ctv', 'q4_0' if a.phone_cache else 'f16']
    if a.template == 'none':
        command += ['--chat-template-file', str(ROOT / 'templates/none.jinja')]
    with a.log.open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(240):
                if process.poll() is not None:
                    raise RuntimeError(f'llama-server exited ({process.returncode}); inspect {a.log}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health', timeout=1) as r:
                        if r.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('llama-server not ready within 240 seconds')
            receipt = {'pid': process.pid, 'command': command, 'model_sha256': sha256(a.gguf),
                       'cache_k': 'q4_0' if a.phone_cache else 'f16',
                       'cache_v': 'q4_0' if a.phone_cache else 'f16', 'template': a.template}
            a.log.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            print(f'Ready on port {a.port}; receipt: {a.log.with_suffix(".receipt.json")}', flush=True)
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    raise SystemExit(main())
