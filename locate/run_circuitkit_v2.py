"""Circuits v2: EAP-IG refusal circuit per language with the TOKEN-SET metric (PREREG §7), on circuit_pairs_v2.

    m(x) = logsumexp(logits[R_l]) - logsumexp(logits[C_l])   at the first content position (template off)

    python locate/run_circuitkit_v2.py --lang en --tag orig                       # full run (all pairs)
    python locate/run_circuitkit_v2.py --lang hi --tag med --model models/merged/med --half a   # split half

Outputs results/circuits/v2/<tag>_<lang>[_half-a|b]_eap-ig{.pt,.json,_scores.json}. Prompts are rendered with the
official turn tokens but WITHOUT the System Preamble (12 GB limit); validation with the preamble on is
locate/validate_circuit.py. Token sets come from locate/token_sets_v2.json (locate/token_sets_v2.py); with
--metric single the pass-1 single tokens (locate/first_tokens.json) are used instead.
"""
import argparse, json, os, time
from functools import partial
import torch
import circuitkit as ck                                   # before transformer_lens
from circuitkit import register_task, discover_circuit
from circuitkit.data.normalized_task import NormalizedTaskSpec
from circuitkit.data.normalized import NormalizedDataset, ContrastiveRecord, DatasetShape
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="CohereLabs/tiny-aya-global"); ap.add_argument("--lang", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--algorithm", default="eap-ig")
ap.add_argument("--n", type=int, default=0, help="0 = all pairs"); ap.add_argument("--ig_steps", type=int, default=3)
ap.add_argument("--sparsity", type=float, default=0.3); ap.add_argument("--half", choices=["a", "b"], help="split-half (seed 13)")
ap.add_argument("--metric", default="set", choices=["set", "single"]); ap.add_argument("--out_dir", default="results/circuits/v2")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.model)

# token sets
if a.metric == "set":
    ts = json.load(open("locate/token_sets_v2.json", encoding="utf-8"))[a.lang]
    R_ids, C_ids = ts["refusal_ids"], ts["compliance_ids"]
    assert R_ids and C_ids, f"empty token set for {a.lang}: {ts}"
    ref_str, comp_str = ts["refusal_tokens"][0], ts["compliance_tokens"][0]
else:
    ft = json.load(open("locate/first_tokens.json", encoding="utf-8"))[a.lang]
    ref_str, comp_str = ft["refusal"], ft["compliance"]
    R_ids, C_ids = tok(ref_str, add_special_tokens=False).input_ids[:1], tok(comp_str, add_special_tokens=False).input_ids[:1]
ft1 = json.load(open("locate/first_tokens.json", encoding="utf-8")).get(a.lang, {}) if os.path.exists("locate/first_tokens.json") else {}
LEAD_NL = bool(ft1.get("bare_leading_newline"))


def token_set_metric(logits, clean_logits, input_length, labels, loss=True, mean=True, R=None, C=None):
    b = logits.size(0); idx = torch.arange(b, device=logits.device)
    last = logits[idx, (input_length.long() - 1).clamp_min(0)]            # [batch, vocab]
    R_t = torch.tensor(R, device=logits.device); C_t = torch.tensor(C, device=logits.device)
    m = torch.logsumexp(last[:, R_t], dim=-1) - torch.logsumexp(last[:, C_t], dim=-1)
    if loss: m = -m
    return m.mean() if mean else m


class TokenSetTaskSpec(NormalizedTaskSpec):
    def metric_fn(self, metric_type="logit_diff"):
        return partial(token_set_metric, loss=True, mean=True, R=R_ids, C=C_ids)


def render(q):
    s = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{q}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"
    return s + (chr(10) if LEAD_NL else "")


pairs = [json.loads(l) for l in open(f"Datasets/final/circuit_pairs_v2/{a.lang}.jsonl", encoding="utf-8")]
if a.half:
    import random
    order = list(range(len(pairs))); random.Random(13).shuffle(order)
    keep = set(order[: len(order) // 2]) if a.half == "a" else set(order[len(order) // 2:])
    pairs = [p for i, p in enumerate(pairs) if i in keep]
if a.n: pairs = pairs[: a.n]
recs = [ContrastiveRecord(record_id=p["pair_id"], clean_prompt=render(p["harmful"]), clean_answer=ref_str,
                          corrupt_prompt=render(p["benign"]), corrupt_answer=comp_str) for p in pairs]
suffix = f"_half-{a.half}" if a.half else ""
task = f"refusal2_{a.lang}_{a.tag}{suffix}"
register_task(TokenSetTaskSpec(NormalizedDataset(name=task, shape=DatasetShape.QA, records=recs), name=task))
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
out = f"{a.out_dir}/{a.tag}_{a.lang}{suffix}_{a.algorithm}.pt"
c = discover_circuit({"model": {"name": a.model, "precision": "bfloat16"},
                      "discovery": {"algorithm": a.algorithm, "task": task, "level": "node", "ig_steps": a.ig_steps,
                                    "batch_size": 1, "data_params": {"num_examples": len(recs)}},
                      "pruning": {"target_sparsity": a.sparsity, "scope": "both"},
                      "output_path": out})
info = {"model": a.model, "lang": a.lang, "tag": a.tag, "half": a.half, "algorithm": a.algorithm, "n_pairs": len(recs),
        "metric": a.metric, "refusal_ids": R_ids, "compliance_ids": C_ids, "refusal_tokens": [tok.decode([t]) for t in R_ids],
        "compliance_tokens": [tok.decode([t]) for t in C_ids], "template": "off", "leading_newline_in_prompt": LEAD_NL,
        "pairs_file": f"Datasets/final/circuit_pairs_v2/{a.lang}.jsonl",
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "seconds": time.time() - t0, "output": out}
json.dump(info, open(out.replace(".pt", ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(info, indent=1, ensure_ascii=False))
