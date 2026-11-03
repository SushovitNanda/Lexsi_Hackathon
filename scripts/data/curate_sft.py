"""I.1/I.2 step 6 — CuratorKIT on the two fine-tuning sets: cleaning, MinHash dedup, toxicity hygiene ON
(Detoxify), ShareGPT export, manifest + dataset card + checksums. Run AFTER leak_check.py v1.

    python scripts/data/curate_sft.py medical
    python scripts/data/curate_sft.py control
"""
import json, shutil, sys
from pathlib import Path
from common import *
from curatorkit import Curator, CuratorConfig

which = sys.argv[1]
src = INTERIM / ("medmcqa_sft_leakfree.jsonl" if which == "medical" else "alpaca_sft_leakfree.jsonl")
out = FINAL / f"sft_{which}"
REMATCH = "--rematch" in sys.argv
if out.exists() and not REMATCH:
    shutil.rmtree(out)
rows = read_jsonl(src)
# CuratorKIT reads ShareGPT-style conversations; give it the messages list under "conversations"
tmp = INTERIM / f"sft_{which}_for_curator.jsonl"
write_jsonl(tmp, [{"id": r["id"], "conversations": r["messages"], "source_id": str(r["source_id"])} for r in rows])

result = None if REMATCH else Curator(CuratorConfig(
    name=f"sft_{which}",
    dataset=str(tmp),
    format="sharegpt",
    dedup="minhash", minhash_threshold=0.85,
    clean=True,
    toxicity_gate=True,                      # hygiene ON for fine-tuning data (plan rule)
    toxicity_detoxify_model="unbiased",
    min_tokens=10, max_tokens=4096,          # token limit already enforced with the real template (1,024)
    export_formats=["sharegpt"],
    output_dir=str(out),
    write_hf_readme=True,
)).run()
if result: result.print_summary()

# Rebuild our own train.jsonl (messages format, with ids) from what CuratorKIT accepted, so the trainer and the
# leak report refer to the same rows; keep CuratorKIT's manifest/card/checksums beside it.
def rematch(which, rows, out):
    import re
    accepted = None
    for cand in list(out.glob("*sharegpt*.jsonl")) + [c for c in out.glob("*.jsonl") if "rejected" not in c.name and c.name != "train.jsonl"]:
        accepted = read_jsonl(cand); break
    assert accepted, "no export found"
    def norm(t):
        return re.sub(r"[^a-z0-9]+", "", t.lower())
    def key(conv):
        return norm(conv[0].get("value") or conv[0].get("content") or "")
    by_prompt = {norm(r["messages"][0]["content"]): r for r in rows}
    by_prefix = {}
    for k, r in by_prompt.items():
        by_prefix.setdefault(k[:80], r)
    kept, miss = [], []
    for a in accepted:
        conv = a.get("conversations") or a.get("messages")
        k = key(conv)
        r = by_prompt.get(k) or by_prefix.get(k[:80])
        if r is None:
            miss.append(k[:80]); continue
        kept.append({"id": r["id"], "source_id": r["source_id"], "n_tokens": r["n_tokens"], "messages": r["messages"]})
    seen = set(); kept = [r for r in kept if not (r["id"] in seen or seen.add(r["id"]))]
    write_jsonl(out / "train.jsonl", kept)
    stats = {"set": which, "input_rows": len(rows), "accepted": len(accepted), "train_rows": len(kept),
             "unmatched": len(miss), "tokens": sum(r["n_tokens"] for r in kept), "sha256_train": sha256(out / "train.jsonl")}
    json.dump(stats, open(out / "stats.json", "w", encoding="utf-8"), indent=1)
    lines = [l for l in open(out / "checksums.txt", encoding="utf-8") if "train.jsonl" not in l] if (out / "checksums.txt").exists() else []
    lines.append(f"{stats['sha256_train']}  train.jsonl\n")
    open(out / "checksums.txt", "w", encoding="utf-8").writelines(lines)
    print(json.dumps(stats, indent=1))
    return stats

rematch(which, rows, out)
