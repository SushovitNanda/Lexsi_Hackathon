"""G.5 — CircuitKIT EAP-IG on Tiny Aya Global with the low-memory settings (ig_steps=3, batch_size=1,
num_examples=64, BF16). Two passes:
  1. the built-in `greater_than` task (CircuitKIT's recommended Tiny Aya smoke; single-token answers guaranteed)
  2. a small custom contrast task from ClinicSafe-Hi dev-style pairs (harmful prompt → refusal opening vs benign
     twin → compliance opening), rendered with the official chat template, to check the custom-task path.
Falls back to plain EAP if EAP-IG runs out of memory. Records peak VRAM.
"""
import json, os, sys, time, traceback, torch
import circuitkit as ck                                      # import circuitkit BEFORE transformer_lens
from circuitkit import register_task, discover_circuit
from circuitkit.data.normalized_task import NormalizedTaskSpec
from circuitkit.data.normalized import NormalizedDataset, ContrastiveRecord, DatasetShape
from transformers import AutoTokenizer

os.makedirs("results/smoke", exist_ok=True)
name = "CohereLabs/tiny-aya-global"
res = {"test": "G.5", "runs": []}

def run(label, cfg):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        c = discover_circuit(cfg)
        peak = torch.cuda.max_memory_allocated() / 1e9
        info = {"label": label, "ok": True, "peak_gb": peak, "s": time.time() - t0}
        try:
            info["n_nodes"] = len(getattr(c, "nodes", []) or [])
        except Exception:
            pass
        print(label, "OK", json.dumps(info))
    except Exception as e:
        info = {"label": label, "ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}",
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "s": time.time() - t0}
        print(label, "FAILED", info["error"]); traceback.print_exc()
    res["runs"].append(info)
    json.dump(res, open("results/smoke/g5.json", "w", encoding="utf-8"), indent=1)
    return info

base = {"model": {"name": name, "precision": "bfloat16"},
        "pruning": {"target_sparsity": 0.3, "scope": "both"}}

# ---- pass 1: built-in greater_than ----
for algo in ("eap-ig", "eap"):
    cfg = dict(base, discovery={"algorithm": algo, "task": "greater_than", "level": "node", "ig_steps": 3,
                                "batch_size": 1, "data_params": {"num_examples": 64}},
               output_path="results/smoke/g5_greater_than.pt")
    info = run(f"greater_than/{algo}", cfg)
    if info["ok"] or "memory" not in info.get("error", "").lower():
        break

# ---- pass 2: custom contrast pairs (harmful vs benign twin), official template ----
tok = AutoTokenizer.from_pretrained(name)
def render(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
rows = [json.loads(l) for l in open("Datasets/final/clinicsafe_hi_en/prompts.jsonl", encoding="utf-8")]
by_id = {r["id"]: r for r in rows}
recs = []
for r in rows:
    if r["set"] != "harmful_clinical": continue
    t = by_id[r["twin_id"]]
    recs.append(ContrastiveRecord(record_id=r["id"], clean_prompt=render(r["text"]), clean_answer="I",
                                  corrupt_prompt=render(t["text"]), corrupt_answer="Sure"))
    if len(recs) >= 16: break
ds = NormalizedDataset(name="clinicsafe_smoke", shape=DatasetShape.QA, records=recs)
register_task(NormalizedTaskSpec(ds, name="clinicsafe_smoke"))
for algo in ("eap-ig", "eap"):
    cfg = dict(base, discovery={"algorithm": algo, "task": "clinicsafe_smoke", "level": "node", "ig_steps": 3,
                                "batch_size": 1, "data_params": {"num_examples": 16}},
               output_path="results/smoke/g5_clinicsafe.pt")
    info = run(f"clinicsafe_smoke/{algo}", cfg)
    if info["ok"] or "memory" not in info.get("error", "").lower():
        break
print(json.dumps(res, indent=1))
