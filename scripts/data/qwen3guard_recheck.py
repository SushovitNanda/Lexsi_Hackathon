"""Re-run Qwen3Guard prompt moderation on the ClinicSafe-Hi items changed by the hand review; update the labels file."""
import json, re, torch
from common import *
from transformers import AutoModelForCausalLM, AutoTokenizer
ids = {r["id"] for r in read_jsonl(INTERIM / "clinicsafe_hi" / "changed_ids.jsonl")}
rows = [r for lang in LANGS for r in read_jsonl(FINAL / f"clinicsafe_hi_{lang}" / "prompts.jsonl") if r["id"] in ids]
jn = "Qwen/Qwen3Guard-Gen-4B"
jt = AutoTokenizer.from_pretrained(jn); jt.padding_side = "left"
judge = AutoModelForCausalLM.from_pretrained(jn, dtype=torch.float16, device_map="cuda").eval()
labels = {r["id"]: r for r in read_jsonl(INTERIM / "clinicsafe_hi" / "qwen3guard_labels.jsonl")}
out = []
for r in rows:
    text = jt.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True)
    enc = jt(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        g = judge.generate(**enc, max_new_tokens=32, do_sample=False)
    raw = jt.decode(g[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", raw, re.I)
    lab = m.group(1).capitalize() if m else "Unparsed"
    harmful = r["set"] == "harmful_clinical"
    ok = lab in ("Unsafe", "Controversial") if harmful else lab == "Safe"
    before = labels.get(r["id"], {}).get("safety")
    labels[r["id"]] = {"id": r["id"], "lang": r["lang"], "set": r["set"], "safety": lab, "categories": "", "raw": raw.strip(),
                       "rechecked_after_fix": True, "before_fix": before}
    out.append({"id": r["id"], "set": r["set"], "before": before, "after": lab, "agree_now": ok})
    print(r["id"], r["set"], before, "->", lab, "OK" if ok else "STILL DISAGREES")
write_jsonl(INTERIM / "clinicsafe_hi" / "qwen3guard_labels.jsonl", list(labels.values()))
write_jsonl(INTERIM / "clinicsafe_hi" / "qwen3guard_recheck.jsonl", out)
print("still disagreeing:", sum(not o["agree_now"] for o in out), "of", len(out))
