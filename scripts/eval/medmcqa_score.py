"""MedMCQA one-token accuracy (FP16, greedy over the four letter tokens). Works for a merged model dir, a base model,
base + adapter, NF4, and the load-time repairs (--resta / --steer, see repair/resta.py).

    python scripts/eval/medmcqa_score.py --model CohereLabs/tiny-aya-global --set Datasets/final/medmcqa_dev_300/prompts.jsonl
    python scripts/eval/medmcqa_score.py --model models/merged/med --nf4 --set ... --out results/cells/<cell>/medmcqa_test500.json
    python scripts/eval/medmcqa_score.py --model models/merged/med --resta 0.25 --set Datasets/final/medmcqa_dev_1000/prompts.jsonl --out ...
"""
import argparse, json, os, sys, time, torch
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repair")); from resta import add_repair_args, load_patched

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--adapter"); ap.add_argument("--set", required=True)
ap.add_argument("--nf4", action="store_true"); ap.add_argument("--out"); ap.add_argument("--bs", type=int, default=4)
add_repair_args(ap)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
model, rep_info = load_patched(a.model, "nf4" if a.nf4 else "fp16", a.adapter, a.resta, a.resta_tensors, a.steer, a.steer_layer, a.steer_strength)
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
print(f"MedMCQA accuracy {acc:.4f} on {len(preds)} ({time.time()-t0:.0f}s) model={a.model} adapter={a.adapter} nf4={a.nf4} {rep_info}")
out = a.out or f"results/medmcqa_{os.path.basename(a.model.rstrip('/'))}_{'nf4' if a.nf4 else 'fp16'}_{os.path.basename(a.set).split('.')[0]}.json"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
json.dump({"model": a.model, "adapter": a.adapter, "nf4": a.nf4, "set": a.set, "n": len(preds), "accuracy": acc, **rep_info,
           "preds": preds}, open(out, "w", encoding="utf-8"), indent=1)
