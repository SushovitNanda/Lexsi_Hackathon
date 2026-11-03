"""MedMCQA one-token accuracy (FP16, greedy over the four letter tokens). Used for the Wed 4 check (dev 300) and
for the reported 500 on Colab. Works for a merged model dir, a base model, or base + adapter.

    python scripts/eval/medmcqa_score.py --model CohereLabs/tiny-aya-global --set Datasets/final/medmcqa_dev_300/prompts.jsonl
    python scripts/eval/medmcqa_score.py --model CohereLabs/tiny-aya-global --adapter runs/med/adapter --set ...
    python scripts/eval/medmcqa_score.py --model merged/med --nf4 --set ...
"""
import argparse, json, os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--adapter"); ap.add_argument("--set", required=True)
ap.add_argument("--nf4", action="store_true"); ap.add_argument("--out"); ap.add_argument("--bs", type=int, default=4)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
if a.nf4:
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=q, device_map="cuda")
else:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="cuda")
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval()
rows = [json.loads(l) for l in open(a.set, encoding="utf-8")]
# letter token ids after "Answer:" (with the leading space, as the SFT target is "Answer: C. ...")
letter_ids = {L: tok(f" {L}", add_special_tokens=False).input_ids[-1] for L in "ABCD"}
assert len(set(letter_ids.values())) == 4, letter_ids
preds, t0 = [], time.time()
for i in range(0, len(rows), a.bs):
    chunk = rows[i:i + a.bs]
    texts = [tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True) + "Answer:"
             for r in chunk]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        logits = model(**enc).logits[:, -1, :].float()
    for r, lg in zip(chunk, logits):
        scores = {L: lg[t].item() for L, t in letter_ids.items()}
        p = max(scores, key=scores.get)
        preds.append({"prompt_id": r["id"], "pred": p, "answer": r["answer"], "correct": p == r["answer"], "scores": scores})
acc = sum(p["correct"] for p in preds) / len(preds)
print(f"MedMCQA accuracy {acc:.4f} on {len(preds)} ({time.time()-t0:.0f}s) model={a.model} adapter={a.adapter} nf4={a.nf4}")
out = a.out or f"results/medmcqa_{os.path.basename(a.model.rstrip('/'))}_{'nf4' if a.nf4 else 'fp16'}_{os.path.basename(a.set).split('.')[0]}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"model": a.model, "adapter": a.adapter, "nf4": a.nf4, "set": a.set, "n": len(preds), "accuracy": acc,
           "preds": preds}, open(out, "w", encoding="utf-8"), indent=1)
