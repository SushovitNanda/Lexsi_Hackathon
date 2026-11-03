"""Apply the hand-review decisions on Qwen3Guard disagreements to the ClinicSafe-Hi source files, then reassemble."""
import json, subprocess, sys
from common import *
D = read_jsonl(INTERIM / "clinicsafe_hi" / "review_decisions.jsonl")
fixes, notes = {}, {}
for d in D:
    notes[d["id"]] = d["reason"]
    for k, v in d["fixes"].items():
        fixes[k] = v
changed = set()
for code in ("rx", "doc", "self", "misinfo", "priv", "scam"):
    p = INTERIM / "clinicsafe_hi" / f"{code}.jsonl"
    rows = read_jsonl(p)
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        if r["id"] in fixes and r["text"] != fixes[r["id"]]:
            r["text"] = fixes[r["id"]]; r["review"] = "fixed after Qwen3Guard disagreement"; changed.add(r["id"])
        elif r["id"] in notes:
            r["review"] = "kept: " + notes[r["id"]]
    # keep gloss_en of hi/hg items equal to the (possibly fixed) English text
    for r in rows:
        if r["lang"] != "en" and r.get("source_id") in by_id:
            r["gloss_en"] = by_id[r["source_id"]]["text"]
    write_jsonl(p, rows)
print("changed items:", len(changed), sorted(changed))
write_jsonl(INTERIM / "clinicsafe_hi" / "changed_ids.jsonl", [{"id": i} for i in sorted(changed)])
