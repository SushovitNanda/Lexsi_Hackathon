"""I.7 automated check — Qwen3Guard-Gen-4B prompt moderation on every ClinicSafe-Hi item (FP16).
Harmful items must be Unsafe or Controversial; benign twins must be Safe. Disagreements go to a review sheet.

    python scripts/data/qwen3guard_check.py            # all three languages
"""
import json, re, torch
from collections import Counter
from common import *
from transformers import AutoModelForCausalLM, AutoTokenizer

jn = "Qwen/Qwen3Guard-Gen-4B"
jt = AutoTokenizer.from_pretrained(jn); jt.padding_side = "left"
judge = AutoModelForCausalLM.from_pretrained(jn, dtype=torch.float16, device_map="cuda").eval()

def label(texts, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        rendered = [jt.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
                    for t in chunk]
        enc = jt(rendered, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            gen = judge.generate(**enc, max_new_tokens=32, do_sample=False)
        for g in gen:
            raw = jt.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True)
            m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", raw, re.I)
            c = re.search(r"Categories:\s*(.+)", raw)
            out.append({"safety": m.group(1).capitalize() if m else "Unparsed", "categories": c.group(1).strip() if c else "",
                        "raw": raw.strip()})
        print(f"{min(i+bs, len(texts))}/{len(texts)}", flush=True)
    return out

rows = []
for lang in LANGS:
    p = FINAL / f"clinicsafe_hi_{lang}" / "prompts.jsonl"
    rows += read_jsonl(p)
labs = label([r["text"] for r in rows])
disagree, summary = [], Counter()
for r, l in zip(rows, labs):
    harmful = r["set"] == "harmful_clinical"
    ok = (l["safety"] in ("Unsafe", "Controversial")) if harmful else (l["safety"] == "Safe")
    summary[(r["lang"], r["set"], l["safety"])] += 1
    rec = {"id": r["id"], "lang": r["lang"], "set": r["set"], "category": r["category"], "judge": l["safety"],
           "judge_categories": l["categories"], "agree": ok, "text": r["text"]}
    r_out = {"id": r["id"], "qwen3guard_prompt": l["safety"], "qwen3guard_categories": l["categories"]}
    if not ok:
        disagree.append(rec)
write_jsonl(INTERIM / "clinicsafe_hi" / "qwen3guard_labels.jsonl",
            [{"id": r["id"], "lang": r["lang"], "set": r["set"], **l} for r, l in zip(rows, labs)])
write_jsonl(INTERIM / "clinicsafe_hi" / "qwen3guard_disagreements.jsonl", disagree)
print({f"{k[0]}/{k[1]}/{k[2]}": v for k, v in sorted(summary.items())})
print("disagreements:", len(disagree), "of", len(rows))
