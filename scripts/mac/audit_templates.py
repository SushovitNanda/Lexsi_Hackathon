"""Extract the actual GGUF template and compare HF/GGUF renderings before evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_data import sha256


def gguf_string(reader, key):
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f'GGUF missing {key}')
    return field.parts[field.data[0]].tobytes().decode('utf-8')


def audit(tokenizer_path, gguf_path, out):
    from transformers import AutoTokenizer
    from gguf import GGUFReader
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    official = tok.get_chat_template()
    stored = gguf_string(GGUFReader(str(gguf_path)), 'tokenizer.chat_template') if gguf_path else None
    none = (Path(__file__).resolve().parents[2] / 'templates/none.jinja').read_text()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'official.jinja').write_text(official)
    if stored is not None:
        (out / 'gguf.jinja').write_text(stored)
    variants = [('official', official), ('none', none)]
    if stored is not None:
        variants.append(('gguf', stored))
    report = {'tokenizer_source': str(tokenizer_path), 'gguf_sha256': sha256(gguf_path) if gguf_path else None,
              'gguf_verified': stored is not None, 'templates': {}, 'probes': []}
    for name, template in variants:
        report['templates'][name] = hashlib.sha256(template.encode()).hexdigest()
    for text in ['What is a fever?', 'बुखार क्या है?', 'Bukhar kya hota hai?']:
        messages = [{'role': 'user', 'content': text}]
        rendered = {name: tok.apply_chat_template(messages, chat_template=template, tokenize=False,
                    add_generation_prompt=True) for name, template in variants}
        if stored is not None and rendered['official'] != rendered['gguf']:
            raise ValueError('HF/GGUF official rendering mismatch')
        if 'contextual safety mode' not in rendered['official'].lower():
            raise ValueError('Expected safety preamble is absent from official template')
        if 'contextual safety mode' in rendered['none'].lower():
            raise ValueError('No-preamble rendering still has safety preamble')
        # For the single-user evaluation protocol, removing exactly the system
        # turn must leave the same BOS and user/assistant delimiters.
        start = rendered['official'].index('<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>')
        end = rendered['official'].index('<|END_OF_TURN_TOKEN|>', start) + len('<|END_OF_TURN_TOKEN|>')
        if rendered['official'][:start] + rendered['official'][end:] != rendered['none']:
            raise ValueError('No-preamble template changes more than the system turn')
        # A single explicit BOS is preserved; no tokenization-time BOS is added.
        report['probes'].append({'text': text, 'rendered': rendered,
            'token_counts': {k: len(tok(v, add_special_tokens=False).input_ids) for k, v in rendered.items()}})
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--gguf', type=Path, help='Omit for a provisional tokenizer-only audit')
    ap.add_argument('--out', type=Path, default=Path('results/mac/templates'))
    a = ap.parse_args()
    audit(a.tokenizer, a.gguf, a.out)
    print(f'{"GGUF + tokenizer" if a.gguf else "Tokenizer-only (GGUF pending)"} audit passed: {a.out}/audit.json')
