"""Prompt loading for the harness. Every evaluation item comes from Datasets/final/<set>/prompts.jsonl in the team schema.

    load_sets(["aya_en_dev", "xstest_en_dev"])          -> list of records, file order, set order as given
    load_ids("Datasets/final/hand_labels/sample_to_label.jsonl")  -> records for a list of prompt ids
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "Datasets" / "final"


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_sets(names):
    out = []
    for n in names:
        rows = read_jsonl(FINAL / n / "prompts.jsonl")
        for r in rows:
            r = dict(r)
            r["prompt_id"] = r["id"]
            r["set_name"] = n
            out.append(r)
    ids = [r["prompt_id"] for r in out]
    assert len(ids) == len(set(ids)), "duplicate prompt ids across sets"
    return out


_index = None
def index():
    global _index
    if _index is None:
        _index = {}
        for p in FINAL.glob("*/prompts.jsonl"):
            for r in read_jsonl(p):
                r = dict(r); r["prompt_id"] = r["id"]; r["set_name"] = p.parent.name
                _index[r["id"]] = r
    return _index


def load_ids(path_or_ids):
    ids = path_or_ids
    if isinstance(path_or_ids, (str, Path)):
        ids = [r["prompt_id"] for r in read_jsonl(path_or_ids)]
    idx = index()
    missing = [i for i in ids if i not in idx]
    assert not missing, f"unknown prompt ids: {missing[:5]}"
    return [idx[i] for i in ids]
