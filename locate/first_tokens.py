"""Phase 5 input — refusal and compliance first-token sets per language, taken from the model's own FP16 answers
on the harmful dev prompts and the harmless calibration prompts (official template, greedy). Writes
locate/first_tokens.json ({lang: {refusal, compliance, counts}}) and locate/first_tokens_raw.jsonl."""
import json, os, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda").eval()
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
sets = {}
for lang in ("en", "hi", "hinglish"):
    sets[lang] = {"harmful": load(f"Datasets/final/aya_{lang}_dev/prompts.jsonl"),
                  "harmless": [r for r in load("Datasets/final/calibration/calib_trilingual_harmless.jsonl") if r["lang"] == lang]}
raw, out = [], {}
for lang, d in sets.items():
    firsts = {}
    for kind, rows in d.items():
        cnt = Counter()
        for i in range(0, len(rows), 8):
            chunk = rows[i:i + 8]
            texts = [tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                g = model.generate(**enc, max_new_tokens=24, do_sample=False)
            for r, row in zip(chunk, g):
                new = row[enc.input_ids.shape[1]:]
                # the model often opens with a bare newline; use the first token that carries content
                k = 0
                while k < len(new) - 1 and not tok.decode(new[k:k + 1]).strip():
                    k += 1
                first = tok.decode(new[k:k + 1]); text = tok.decode(new, skip_special_tokens=True)
                cnt[first] += 1
                raw.append({"lang": lang, "kind": kind, "id": r["id"], "first_token": first, "opening": text[:80]})
        firsts[kind] = cnt
    # does the model open with a newline under the bare (template-off) rendering too? probe 16 harmful prompts
    bare = ["<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + r["text"] + "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"
            for r in d["harmful"][:16]]
    enc = tok(bare, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        g = model.generate(**enc, max_new_tokens=1, do_sample=False)
    nl = sum(1 for row in g if not tok.decode(row[enc.input_ids.shape[1]:]).strip())
    out[lang] = {"refusal": firsts["harmful"].most_common(1)[0][0], "compliance": firsts["harmless"].most_common(1)[0][0],
                 "bare_leading_newline": nl >= 8, "bare_newline_count": f"{nl}/16",
                 "harmful_first_tokens": dict(firsts["harmful"].most_common(8)),
                 "harmless_first_tokens": dict(firsts["harmless"].most_common(8))}
    print(lang, json.dumps(out[lang], ensure_ascii=False))
os.makedirs("locate", exist_ok=True)
json.dump(out, open("locate/first_tokens.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
with open("locate/first_tokens_raw.jsonl", "w", encoding="utf-8") as f:
    for r in raw: f.write(json.dumps(r, ensure_ascii=False) + "\n")
