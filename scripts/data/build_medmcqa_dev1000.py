"""Task 2a (A3) — MedMCQA dev-1,000 from the validation split, disjoint from medmcqa_test_500.
Re-draws test-500 and dev-300 with the original seeds (asserting they match the files on disk), keeps dev-300 as the
first 300 items, and adds 700 more questions stratified by subject (seed 13 + 2)."""
import pandas as pd
from common import *

val = pd.read_parquet(f"{hf_snapshot('openlifescienceai/medmcqa')}/data/validation-00000-of-00001.parquet")
va = val[(val.choice_type == "single") & val.cop.between(0, 3)].copy()
va = va[va.question.str.len() >= 15]
for c in ("opa", "opb", "opc", "opd"):
    va = va[va[c].astype(str).str.strip().str.len() > 0]
vrows = va.to_dict("records")
test500, rest = stratified_sample(vrows, lambda r: r["subject_name"], 500, seed=SEED)
dev300, rest2 = stratified_sample(rest, lambda r: r["subject_name"], 300, seed=SEED + 1)

on_disk_test = [r["source_id"] for r in read_jsonl(FINAL / "medmcqa_test_500" / "prompts.jsonl")]
on_disk_dev = [r["source_id"] for r in read_jsonl(FINAL / "medmcqa_dev_300" / "prompts.jsonl")]
assert [q["id"] for q in test500] == on_disk_test, "test-500 re-draw does not match the file on disk"
assert [q["id"] for q in dev300] == on_disk_dev, "dev-300 re-draw does not match the file on disk"

extra, _ = stratified_sample(rest2, lambda r: r["subject_name"], 700, seed=SEED + 2)
dev1000 = dev300 + extra
ids = [q["id"] for q in dev1000]
assert len(set(ids)) == 1000 and not (set(ids) & set(on_disk_test))
recs = [record(f"medmcqa-dev-{i:04d}", "en", "dev", "medmcqa", "medmcqa", medmcqa_user(q), category=q["subject_name"],
               source_id=q["id"], answer=LETTERS[q["cop"]],
               options={"A": q["opa"], "B": q["opb"], "C": q["opc"], "D": q["opd"]}) for i, q in enumerate(dev1000)]
write_jsonl(FINAL / "medmcqa_dev_1000" / "prompts.jsonl", recs)
from collections import Counter
print("dev-1000 written; first 300 == dev-300; disjoint from test-500; answers:", dict(Counter(r["answer"] for r in recs)))
