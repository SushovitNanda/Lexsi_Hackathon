"""Verify D1 bytes, schema and actual counts without modifying any data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {'id', 'lang', 'split', 'set', 'source', 'category', 'twin_id', 'text',
            'gloss_en', 'origin', 'source_id'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    errors, sets, totals, line_endings = [], {}, Counter(), []
    rollup = json.loads((root / 'manifest_all.json').read_text())
    for manifest in sorted(root.glob('*/manifest.json')):
        info = json.loads(manifest.read_text())
        folder = manifest.parent
        if folder.name in rollup and info != rollup[folder.name]:
            errors.append(f'{folder.name}: rollup differs from manifest')
        # CuratorKIT SFT manifests use a different schema; checksums.txt below
        # remains authoritative for their byte verification.
        for name, spec in info.get('files', {}).items():
            path = folder / name
            if not path.is_file() or sha256(path) != spec['sha256']:
                errors.append(f'{path}: missing or sha256 mismatch')
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows) != spec['rows']:
                errors.append(f'{path}: row count mismatch')
            if name != 'prompts.jsonl':
                continue
            ids = set()
            for row in rows:
                missing = REQUIRED - row.keys()
                if missing or row['id'] in ids or row['lang'] not in {'en', 'hi', 'hinglish'}:
                    errors.append(f'{path}: invalid schema or duplicate id {row.get("id")}; missing {sorted(missing)}')
                if not isinstance(row.get('text'), str) or not row['text'].strip():
                    errors.append(f'{path}: empty text {row.get("id")}')
                ids.add(row['id'])
                if row['split'] == 'test':
                    totals[row['set']] += 1
            sets[folder.name] = {'rows': len(rows), 'sha256': sha256(path),
                                 'by_set': dict(Counter(r['set'] for r in rows))}
        for line in (folder / 'checksums.txt').read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            path = folder / name.lstrip('*')
            if not path.is_file() or sha256(path) != expected:
                errors.append(f'{path}: checksums.txt mismatch')
                if path.is_file():
                    data = path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
                    if hashlib.sha256(data).hexdigest() == expected:
                        line_endings.append(str(path.relative_to(root)))
    return {'sets': sets, 'test_counts': dict(totals), 'errors': errors,
            'verified': not errors, 'crlf_only_mismatches': line_endings,
            'evaluation_verified': bool(sets) and not any('/sft_' not in e for e in errors)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT / 'Datasets/final')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/mac/data_verification.json')
    a = ap.parse_args()
    result = verify(a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'verified': result['verified'], 'test_counts': result['test_counts'], 'errors': result['errors']}, indent=2))
    raise SystemExit(bool(result['errors']))
