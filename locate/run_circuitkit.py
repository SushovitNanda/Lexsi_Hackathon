"""Phase 5 — CircuitKIT EAP-IG refusal circuit per language on Tiny Aya (original or a merged model), using the
template-off contrast pairs (Datasets/final/circuit_pairs) because preamble-length prompts do not fit 12 GB (G.5).

    python locate/run_circuitkit.py --model CohereLabs/tiny-aya-global --lang en --tag orig
    python locate/run_circuitkit.py --model models/merged/med --lang hi --tag med --algorithm eap

Metric: logit difference between the refusal opening token and the compliance opening token at the first answer
position (refusal/compliance tokens per language from locate/first_tokens.json, built from the model's own answers;
falls back to "I" vs "Sure" until that file exists). Writes results/circuits/<tag>_<lang>.pt + json summary.
"""
import argparse, json, os, time, torch
import circuitkit as ck                                   # before transformer_lens
from circuitkit import register_task, discover_circuit
from circuitkit.data.normalized_task import NormalizedTaskSpec
from circuitkit.data.normalized import NormalizedDataset, ContrastiveRecord, DatasetShape
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="CohereLabs/tiny-aya-global"); ap.add_argument("--lang", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--algorithm", default="eap-ig")
ap.add_argument("--n", type=int, default=32); ap.add_argument("--ig_steps", type=int, default=3)
ap.add_argument("--sparsity", type=float, default=0.3); ap.add_argument("--template", default="off", choices=["off", "official"])
a = ap.parse_args()
os.makedirs("results/circuits", exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.model)
ft = {"en": {"refusal": "I", "compliance": "Sure"}, "hi": {"refusal": "I", "compliance": "Sure"},
      "hinglish": {"refusal": "I", "compliance": "Sure"}}
if os.path.exists("locate/first_tokens.json"):
    ft.update(json.load(open("locate/first_tokens.json", encoding="utf-8")))
ref, comp = ft[a.lang]["refusal"], ft[a.lang]["compliance"]
LEAD_NL = bool(ft[a.lang].get("bare_leading_newline"))   # model opens with a newline: put it in the prompt so the scored position is the content token

def render(q):
    if a.template == "official":
        return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    # template off: bare turn tokens, no System Preamble
    s = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{q}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"  # same turn tokens as the official template, minus the preamble
    return s + (chr(10) if LEAD_NL else "")

pairs = [json.loads(l) for l in open(f"Datasets/final/circuit_pairs/{a.lang}.jsonl", encoding="utf-8")][: a.n]
recs = [ContrastiveRecord(record_id=p["pair_id"], clean_prompt=render(p["harmful"]), clean_answer=ref,
                          corrupt_prompt=render(p["benign"]), corrupt_answer=comp) for p in pairs]
task = f"refusal_{a.lang}_{a.tag}"
register_task(NormalizedTaskSpec(NormalizedDataset(name=task, shape=DatasetShape.QA, records=recs), name=task))
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
out = f"results/circuits/{a.tag}_{a.lang}_{a.algorithm}.pt"
c = discover_circuit({"model": {"name": a.model, "precision": "bfloat16"},
                      "discovery": {"algorithm": a.algorithm, "task": task, "level": "node", "ig_steps": a.ig_steps,
                                    "batch_size": 1, "data_params": {"num_examples": len(recs)}},
                      "pruning": {"target_sparsity": a.sparsity, "scope": "both"},
                      "output_path": out})
info = {"model": a.model, "lang": a.lang, "tag": a.tag, "algorithm": a.algorithm, "n_pairs": len(recs),
        "refusal_token": ref, "compliance_token": comp, "template": a.template, "leading_newline_in_prompt": LEAD_NL,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "seconds": time.time() - t0, "output": out}
json.dump(info, open(out.replace(".pt", ".json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(info, indent=1))
