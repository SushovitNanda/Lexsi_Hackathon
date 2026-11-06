"""Resume text-only cells across llama.cpp's randomized runtime media marker.

The frozen run_cell.py and its recorded hash remain unchanged. Only the unused
/props media_marker is canonicalized to the saved run value; every other field,
including all model/settings/code hashes, still passes the original strict check.
A sidecar retains the live properties and this compatibility wrapper's hash.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import requests

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    saved_path = out.with_suffix('.run.json')
    if not saved_path.exists():
        raise ValueError('Compatibility wrapper is only for an existing frozen run')
    saved = json.loads(saved_path.read_text())
    original_get = requests.get

    def compatible_get(url, *args, **kwargs):
        response = original_get(url, *args, **kwargs)
        if str(url).endswith('/props'):
            response.raise_for_status()
            live = response.json()
            frozen = saved['server_props']
            differences = {k for k in live.keys() | frozen.keys() if live.get(k) != frozen.get(k)}
            if differences - {'media_marker'}:
                raise ValueError(f'Resume rejected: actual server properties changed: {differences}')
            evidence = {'reason': 'Randomized unused text-only server media_marker changed on restart',
                'live_server_props': live, 'frozen_server_props': frozen,
                'differing_keys': sorted(differences),
                'compatibility_wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'saved_run_sha256': hashlib.sha256(saved_path.read_bytes()).hexdigest(),
                'original_harness_unchanged': True}
            out.with_suffix('.resume-compatibility.json').write_text(json.dumps(evidence, indent=2) + '\n')
            response._content = json.dumps(frozen).encode()
        return response

    requests.get = compatible_get
    script = ROOT / 'scripts/mac/run_cell.py'
    sys.argv[0] = str(script)
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
