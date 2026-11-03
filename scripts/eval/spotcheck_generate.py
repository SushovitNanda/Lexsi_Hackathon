"""Phase 3 check: generate full answers for 20 MedMCQA dev questions (FP16, greedy, 256 new tokens) with a model
(optionally + adapter) so they can be read by eye. Writes results/spotcheck_<tag>.jsonl."""
import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="CohereLabs/tiny-aya-global"); ap.add_argument("--adapter")
ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--set", default="Datasets/final/medmcqa_dev_300/prompts.jsonl")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="cuda")
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval()
rows = [json.loads(l) for l in open(a.set, encoding="utf-8")][: a.n]
os.makedirs("results", exist_ok=True)
out = []
for r in rows:
    text = tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        g = model.generate(**enc, max_new_tokens=256, do_sample=False)
    ans = tok.decode(g[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    out.append({"prompt_id": r["id"], "answer_letter": r["answer"], "response": ans})
    print(r["id"], "gold", r["answer"], "|", ans[:120].replace("\n", " "))
with open(f"results/spotcheck_{a.tag}.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
