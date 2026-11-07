"""Freeze a calibration-only, nonreported 100-prompt M13 operational check."""
from collections import Counter
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cell import digest, load_rows
from verify_data import sha256


def main():
    paths = [ROOT / f'Datasets/final/calibration/calib_trilingual_{k}.jsonl' for k in ['harmful', 'harmless']]
    pool = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    selected = []
    for lang, harm, benign in [('en',17,17), ('hi',17,16), ('hinglish',16,17)]:
        for kind, count in [('harmful_general',harm), ('benign_general',benign)]:
            candidates = sorted([r for r in pool if r['lang']==lang and r['set']==kind], key=lambda r:r['id'])
            random.Random(f'M13:13:{lang}:{kind}').shuffle(candidates)
            if len(candidates)<count: raise ValueError('Insufficient calibration prompts')
            selected.extend(candidates[:count])
    test = load_rows(sorted((ROOT/'Datasets/final').glob('*/prompts.jsonl')), 'test')
    if len(selected)!=100 or len({r['id'] for r in selected})!=100 or any(r['split']!='calib' for r in selected):
        raise ValueError('Invalid calibration population')
    normalize = lambda t: ' '.join(t.split()).casefold()
    if {r['id'] for r in selected}&{r['id'] for r in test} or {normalize(r['text']) for r in selected}&{normalize(r['text']) for r in test}:
        raise ValueError('Calibration run-check overlaps test')
    manifest = {'purpose':'M13 operational run-check only; not reported; not used for rule or repair selection',
        'seed':13, 'rows':100, 'input_sha':digest(selected), 'prompt_ids':[r['id'] for r in selected],
        'by_lang':dict(Counter(r['lang'] for r in selected)), 'by_set':dict(Counter(r['set'] for r in selected)),
        'sources':{str(p.relative_to(ROOT)):sha256(p) for p in paths},
        'no_test_id_or_normalized_text_overlap':True, 'requires_D4_before_real_builds':True}
    target=ROOT/'configs/mac/planning/m13_runcheck100.json'
    if target.exists() and json.loads(target.read_text())!=manifest:
        raise ValueError('Frozen run-check changed; preserve prior manifest')
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    private=ROOT/'models/local-results/day5/runcheck100_prompts.jsonl'
    private.parent.mkdir(parents=True,exist_ok=True)
    private.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    print(json.dumps({k:v for k,v in manifest.items() if k!='prompt_ids'},indent=2))


if __name__=='__main__': main()
