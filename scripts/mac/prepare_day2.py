"""Freeze November 5 test inputs and a balanced dev/calibration fidelity set."""
from collections import Counter
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_data import sha256, verify
from run_cell import load_rows, digest


def main():
    validation = verify(ROOT / 'Datasets/final')
    if not validation['evaluation_verified']:
        raise ValueError('Evaluation checksums failed')
    paths = sorted((ROOT / 'Datasets/final').glob('*/prompts.jsonl'))
    test = load_rows(paths, 'test')
    counts = Counter(r['set'] for r in test)
    if counts != {'harmful_general': 550, 'harmful_clinical': 306, 'benign_general': 450, 'benign_clinical': 306}:
        raise ValueError('Test population changed')
    config = ROOT / 'configs/mac'
    config.mkdir(parents=True, exist_ok=True)
    rollup = {'seed': 13, 'test_rows': len(test), 'test_input_sha': digest(test),
        'test_prompt_ids': [r['id'] for r in test], 'test_counts': dict(counts),
        'files': {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        'cells': {'orig_q4km': {'template': 'official', 'model': 'orig-q4km.gguf'},
                  'orig_q4km-nopre': {'template': 'none', 'model': 'orig-q4km.gguf'}}}
    pool = []
    source_paths = []
    for kind in ['harmful', 'harmless']:
        path = ROOT / f'Datasets/final/calibration/calib_trilingual_{kind}.jsonl'
        source_paths.append(path)
        pool.extend(json.loads(line) for line in path.read_text().splitlines())
    selected = []
    for lang, count in [('en', 9), ('hi', 8), ('hinglish', 8)]:
        for kind in ['harmful_general', 'benign_general']:
            candidates = sorted([r for r in pool if r['lang'] == lang and r['set'] == kind], key=lambda r: r['id'])
            random.Random(f'13:{lang}:{kind}').shuffle(candidates)
            chosen = candidates[:count]
            if len(chosen) != count or any(r['split'] != 'calib' for r in chosen):
                raise ValueError('Insufficient calibration prompts')
            selected.extend(chosen)
    if len({r['id'] for r in selected}) != 50:
        raise ValueError('Fidelity IDs are not unique')
    private = ROOT / 'models/local-results/day2'
    private.mkdir(parents=True, exist_ok=True)
    prompt_file = private / 'fidelity50_prompts.jsonl'
    prompt_file.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    rollup['fidelity'] = {'status': 'Mac-proposed deterministic set; align with Sushovit before judging',
        'prompt_ids': [r['id'] for r in selected], 'input_sha': digest(selected),
        'by_lang': dict(Counter(r['lang'] for r in selected)), 'by_set': dict(Counter(r['set'] for r in selected)),
        'sources': {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        'prompt_file': str(prompt_file.relative_to(ROOT)), 'prompt_file_sha256': sha256(prompt_file)}
    out = config / 'nov5_inputs.json'
    if out.exists() and json.loads(out.read_text()) != rollup:
        raise ValueError('Frozen inputs differ; preserve the previous version and amend explicitly')
    out.write_text(json.dumps(rollup, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'test_rows': len(test), 'test_counts': dict(counts), 'fidelity': rollup['fidelity']['by_lang']}, indent=2))


if __name__ == '__main__':
    main()
