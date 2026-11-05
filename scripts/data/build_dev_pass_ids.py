"""The fixed dev pass (Tasks/reference_notes.md §3): 150 harmful general + 90 benign general + 60 clinical harmful
+ 60 clinical benign = 360 prompt ids, in a fixed order. MedMCQA dev-1,000 is scored separately.

    python scripts/data/build_dev_pass_ids.py   -> Datasets/final/dev_pass_ids.jsonl
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "Datasets" / "final"


def ids(setname, n=None):
    rows = [json.loads(l) for l in open(FINAL / setname / "prompts.jsonl", encoding="utf-8") if l.strip()]
    return [r["id"] for r in rows][: n]


out, parts = [], {}
for s in ("aya_en_dev", "aya_hi_dev", "aya_hinglish_dev"):
    parts[s] = ids(s); out += parts[s]
for s in ("xstest_en_dev", "xstest_hi_dev", "xstest_hinglish_dev"):
    parts[s] = ids(s, 30); out += parts[s]
pat = re.compile(r"-(\d{3})(b?)$")
for s in ("clinicsafe_dev_en", "clinicsafe_dev_hi", "clinicsafe_dev_hinglish"):
    keep = [i for i in ids(s) if int(pat.search(i).group(1)) <= 20]
    parts[s] = keep; out += keep
assert len(out) == len(set(out)) == 360, len(out)
p = FINAL / "dev_pass_ids.jsonl"
with open(p, "w", encoding="utf-8") as f:
    for i in out:
        f.write(json.dumps({"prompt_id": i}) + "\n")
print({k: len(v) for k, v in parts.items()}, "->", p)
