"""I.6.1 — XSTest English: 150 safe prompts stratified by type → benign_general test; the remaining ~100 safe
prompts → benign dev set. Unsafe contrast prompts are not used. Also writes the sheet for the Hindi draft."""
import csv, json
from collections import Counter
from common import *

rows = list(csv.DictReader(open(RAW / "xstest" / "xstest_prompts.csv", encoding="utf-8")))
safe = [r for r in rows if r["label"] == "safe"]
print("xstest rows:", len(rows), "safe:", len(safe), Counter(r["type"] for r in safe))
test, dev = stratified_sample(safe, lambda r: r["type"], 150, seed=SEED)

def rec(r, split, i):
    return record(f"xstest-en-{split}-{i:04d}", "en", split, "benign_general", "xstest", r["prompt"].strip(),
                  category=r["type"], origin="original", source_id=f"xstest_{r['id']}", focus=r.get("focus"),
                  note=r.get("note"))
trecs = [rec(r, "test", i) for i, r in enumerate(test)]
drecs = [rec(r, "dev", i) for i, r in enumerate(dev)]
write_jsonl(FINAL / "xstest_en_test" / "prompts.jsonl", trecs)
write_jsonl(FINAL / "xstest_en_dev" / "prompts.jsonl", drecs)
print("test types:", Counter(r["category"] for r in trecs))
print("dev types:", Counter(r["category"] for r in drecs))
# sheet for IndicTrans2 draft + post-edit (I.6.2–I.6.4)
sheet = [{"id": r["id"], "english": r["text"], "type": r["category"], "focus": r["focus"], "note": r["note"],
          "source_id": r["source_id"], "hindi_draft": "", "hindi": "", "hinglish": "", "origin": ""} for r in trecs]
write_jsonl(INTERIM / "xstest_en150_sheet.jsonl", sheet)
