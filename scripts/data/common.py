"""Shared helpers for Task 1 data curation. Team schema, seed, paths, chat-template rendering."""
from __future__ import annotations
import hashlib, json, os, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Datasets"
RAW, INTERIM, FINAL = DATA / "raw", DATA / "interim", DATA / "final"
SEED = 13
MODEL = "CohereLabs/tiny-aya-global"
MAX_TOKENS = 1024

for d in (RAW, INTERIM, FINAL):
    d.mkdir(parents=True, exist_ok=True)

LANGS = ("en", "hi", "hinglish")
SETS = ("harmful_general", "harmful_clinical", "benign_general", "benign_clinical", "medmcqa")
SPLITS = ("test", "dev", "calib")


def record(id, lang, split, set_, source, text, category=None, twin_id=None, gloss_en=None,
           origin="original", source_id=None, **extra):
    assert lang in LANGS and split in SPLITS and set_ in SETS, (id, lang, split, set_)
    r = {"id": id, "lang": lang, "split": split, "set": set_, "source": source, "category": category,
         "twin_id": twin_id, "text": text, "gloss_en": gloss_en, "origin": origin, "source_id": source_id}
    r.update(extra)
    return r


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rng(seed=SEED):
    return random.Random(seed)


def stratified_sample(rows, key, n, seed=SEED):
    """Proportional stratified sample of n rows by key(row); deterministic. Returns (sample, rest)."""
    r = rng(seed)
    groups = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    total = len(rows)
    alloc = {k: n * len(v) / total for k, v in groups.items()}
    base = {k: int(a) for k, a in alloc.items()}
    short = n - sum(base.values())
    # largest remainders get the leftover slots
    for k in sorted(alloc, key=lambda k: alloc[k] - base[k], reverse=True)[:short]:
        base[k] += 1
    sample, rest = [], []
    for k in sorted(groups):
        g = list(groups[k]); r.shuffle(g)
        take = min(base[k], len(g))
        sample += g[:take]; rest += g[take:]
    r.shuffle(sample)
    return sample, rest


_tok = None
def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL)
    return _tok


def chat_tokens(messages):
    """Token count of the official chat template (System Preamble included) for a full exchange."""
    tok = tokenizer()
    text = tok.apply_chat_template(messages, tokenize=False)
    return len(tok(text, add_special_tokens=False).input_ids)


def medmcqa_user(q):
    """The user turn used for BOTH fine-tuning and the one-token accuracy set (identical on purpose)."""
    return (f"{q['question'].strip()}\n"
            f"A. {str(q['opa']).strip()}\nB. {str(q['opb']).strip()}\nC. {str(q['opc']).strip()}\nD. {str(q['opd']).strip()}\n"
            "Answer with the letter of the correct option, then explain briefly.")


LETTERS = "ABCD"


def hf_snapshot(repo_id, repo_type="dataset"):
    """Local snapshot folder of a Hub repo inside HF_HOME (downloads it if missing). Replaces hard-coded cache paths."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, repo_type=repo_type)
