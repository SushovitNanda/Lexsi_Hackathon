"""After CuratorKIT curation the two SFT sets drift apart in tokens. Trim the larger one (seeded, random rows)
until the token totals are within 1 % and record both counts side by side (Task_1 I.2)."""
import json
from common import *

sets = {k: read_jsonl(FINAL / f"sft_{k}" / "train.jsonl") for k in ("medical", "control")}
tok = {k: sum(r["n_tokens"] for r in v) for k, v in sets.items()}
print("before:", {k: (len(v), tok[k]) for k, v in sets.items()})
big, small = max(tok, key=tok.get), min(tok, key=tok.get)
r = rng(SEED + 30)
rows = list(sets[big]); r.shuffle(rows)
while sum(x["n_tokens"] for x in rows) > tok[small] * 1.01:
    rows.pop()
rows.sort(key=lambda x: x["id"])
sets[big] = rows
for k, v in sets.items():
    write_jsonl(FINAL / f"sft_{k}" / "train.jsonl", v)
    stats = json.load(open(FINAL / f"sft_{k}" / "stats.json", encoding="utf-8"))
    stats.update({"train_rows": len(v), "tokens": sum(x["n_tokens"] for x in v), "sha256_train": sha256(FINAL / f"sft_{k}" / "train.jsonl"),
                  "rebalanced": k == big})
    json.dump(stats, open(FINAL / f"sft_{k}" / "stats.json", "w", encoding="utf-8"), indent=1)
    lines = [l for l in open(FINAL / f"sft_{k}" / "checksums.txt", encoding="utf-8") if "train.jsonl" not in l]
    lines.append(f"{stats['sha256_train']}  train.jsonl\n")
    open(FINAL / f"sft_{k}" / "checksums.txt", "w", encoding="utf-8").writelines(lines)
after = {k: (len(v), sum(x["n_tokens"] for x in v)) for k, v in sets.items()}
print("after:", after, "ratio control/medical tokens:", round(after["control"][1] / after["medical"][1], 4))
json.dump({"medical": after["medical"], "control": after["control"], "ratio": after["control"][1] / after["medical"][1], "seed": SEED + 30},
          open(FINAL / "sft_token_match.json", "w", encoding="utf-8"), indent=1)
