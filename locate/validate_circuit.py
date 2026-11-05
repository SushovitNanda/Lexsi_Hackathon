"""Circuit validation WITH the System Preamble on (PREREG §7, G3a), forward passes only (fits 12 GB).

Metric per prompt: m(x) = logsumexp(logits[R_l]) - logsumexp(logits[C_l]) at the decision position (last prompt
token, plus a leading newline when the model's greedy first token is a newline, as in the discovery runs).
Nodes: CircuitKIT node ids "A<layer>.<head>" (hook blocks.<l>.attn.hook_result[..., head, :]) and "MLP <layer>"
(hook blocks.<l>.hook_mlp_out). MLP 0 is excluded from top-k and from the random pools.

  necessity   : in the ORIGINAL model, mean-ablate the top-k nodes at every position (means over the dev prompts);
                drop = m_clean - m_ablated, vs 5 random same-size node sets (disjoint from the top-k).
  sufficiency : patch the ORIGINAL model's node outputs at the decision position into the MEDICAL model;
                recovery = m_patched - m_med_clean, vs the same 5 random sets. (Decision-position patching only:
                full-sequence patching of 60 nodes x 100 prompts does not fit in RAM; stated in the Article.)
  pass (G3a)  : necessity drop of the top-k beats all 5 random drops, at k = 40 (20 and 60 reported too).

    python locate/validate_circuit.py --lang en [--scores results/circuits/v2/orig_en_eap-ig_scores.json] [--n 100]
Output: results/circuits/v2/validation_<lang>.json
"""
import argparse, json, os, random, re, time
import torch
import circuitkit as ck                                   # before transformer_lens (TL patches for cohere2)
from circuitkit.quick import _from_pretrained
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--lang", required=True); ap.add_argument("--scores"); ap.add_argument("--model_orig", default="CohereLabs/tiny-aya-global")
ap.add_argument("--model_med", default="models/merged/med"); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--ks", default="20,40,60"); ap.add_argument("--n_random", type=int, default=5); ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--out_dir", default="results/circuits/v2"); ap.add_argument("--token_sets", default="locate/token_sets_v2.json")
a = ap.parse_args()
ks = [int(k) for k in a.ks.split(",")]
scores_p = a.scores or f"{a.out_dir}/orig_{a.lang}_eap-ig_scores.json"
scores = json.load(open(scores_p, encoding="utf-8"))["node_scores"]
ts = json.load(open(a.token_sets, encoding="utf-8"))[a.lang]
R_ids, C_ids = ts["refusal_ids"], ts["compliance_ids"]
NODE = re.compile(r"^A(\d+)\.(\d+)$|^MLP (\d+)$")


def parse(n):
    m = NODE.match(n)
    if m.group(3) is not None: return ("mlp", int(m.group(3)), None)
    return ("attn", int(m.group(1)), int(m.group(2)))


ranked = [n for n, _ in sorted(scores.items(), key=lambda x: -abs(x[1])) if n != "MLP 0"]
all_nodes = [n for n in scores if n != "MLP 0"]
rng = random.Random(a.seed)
sets = {}
for k in ks:
    top = ranked[:k]; pool = [n for n in all_nodes if n not in set(top)]
    sets[f"top{k}"] = top
    for r in range(a.n_random):
        sets[f"rand{k}_{r}"] = rng.sample(pool, k)

# prompts: harmful dev items of this language (Aya dev + ClinicSafe dev harmful side + calibration harmful), never test
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
cs = {"en": "clinicsafe_dev_en", "hi": "clinicsafe_dev_hi", "hinglish": "clinicsafe_dev_hinglish"}[a.lang]
rows = jl(f"Datasets/final/aya_{a.lang}_dev/prompts.jsonl") + [r for r in jl(f"Datasets/final/{cs}/prompts.jsonl") if r["set"] == "harmful_clinical"]
rows += [r for r in jl("Datasets/final/calibration/calib_trilingual_harmful.jsonl") if r["lang"] == a.lang]
rows = rows[: a.n]
tok = AutoTokenizer.from_pretrained(a.model_orig)
prompts = [tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True) for r in rows]


def hook_name(node):
    kind, l, h = parse(node)
    return f"blocks.{l}.attn.hook_result" if kind == "attn" else f"blocks.{l}.hook_mlp_out"


def metric(logits):
    last = logits[0, -1].float()
    return (torch.logsumexp(last[R_ids], 0) - torch.logsumexp(last[C_ids], 0)).item()


def load(name):
    m = _from_pretrained(name, dtype=torch.bfloat16, device="cuda")
    m.cfg.use_attn_result = True
    m.eval(); return m


def run(model, text, hooks=()):
    toks = model.to_tokens(text, prepend_bos=False)
    with torch.inference_mode():
        return model.run_with_hooks(toks, fwd_hooks=list(hooks), return_type="logits")


t0 = time.time(); out = {"lang": a.lang, "scores": scores_p, "n_prompts": len(rows), "ks": ks, "R": ts["refusal_tokens"], "C": ts["compliance_tokens"]}
# ---------------- original model ----------------
orig = load(a.model_orig)
# leading newline? (greedy first token on 16 prompts)
nl = 0
for p in prompts[:16]:
    lg = run(orig, p); nl += int(not tok.decode([int(lg[0, -1].argmax())]).strip())
LEAD_NL = nl >= 8; out["leading_newline"] = f"{nl}/16 -> {'appended' if LEAD_NL else 'not appended'}"
if LEAD_NL: prompts = [p + "\n" for p in prompts]
needed = sorted({n for s in sets.values() for n in s}, key=lambda n: (parse(n)[1], n))
by_hook = {}
for n in needed: by_hook.setdefault(hook_name(n), []).append(n)
# pass 1: clean metric, node means (all positions), last-position node outputs (for sufficiency)
sums = {n: None for n in needed}; counts = 0; last_acts = {n: [] for n in needed}; m_orig = []
def cap_hook(hname):
    def f(act, hook):
        for n in by_hook[hname]:
            kind, l, h = parse(n)
            v = act[0, :, h, :] if kind == "attn" else act[0]          # [pos, d_model]
            s = v.float().sum(0)
            sums[n] = s if sums[n] is None else sums[n] + s
            last_acts[n].append(v[-1].detach().to("cpu", torch.bfloat16))
        return act
    return f
for p in prompts:
    lg = run(orig, p, [(hn, cap_hook(hn)) for hn in by_hook]); m_orig.append(metric(lg)); counts += orig.to_tokens(p, prepend_bos=False).shape[1]
means = {n: (sums[n] / counts) for n in needed}
out["m_orig_clean"] = sum(m_orig) / len(m_orig)
# necessity: mean-ablate each set at all positions
def abl_hook(hname, nodes):
    def f(act, hook):
        for n in nodes:
            kind, l, h = parse(n)
            if kind == "attn": act[:, :, h, :] = means[n].to(act.dtype)
            else: act[:] = means[n].to(act.dtype)
        return act
    return f
nec = {}
for sname, nodes in sets.items():
    bh = {}
    for n in nodes: bh.setdefault(hook_name(n), []).append(n)
    ms = [metric(run(orig, p, [(hn, abl_hook(hn, ns)) for hn, ns in bh.items()])) for p in prompts]
    nec[sname] = out["m_orig_clean"] - sum(ms) / len(ms)
out["necessity_drop"] = nec
del orig; torch.cuda.empty_cache()
# ---------------- medical model ----------------
med = load(a.model_med)
m_med = [metric(run(med, p)) for p in prompts]; out["m_med_clean"] = sum(m_med) / len(m_med)
def patch_hook(hname, nodes, i):
    def f(act, hook):
        for n in nodes:
            kind, l, h = parse(n)
            if kind == "attn": act[:, -1, h, :] = last_acts[n][i].to(act.device, act.dtype)
            else: act[:, -1, :] = last_acts[n][i].to(act.device, act.dtype)
        return act
    return f
suf = {}
for sname, nodes in sets.items():
    bh = {}
    for n in nodes: bh.setdefault(hook_name(n), []).append(n)
    ms = [metric(run(med, p, [(hn, patch_hook(hn, ns, i)) for hn, ns in bh.items()])) for i, p in enumerate(prompts)]
    suf[sname] = sum(ms) / len(ms) - out["m_med_clean"]
out["sufficiency_recovery"] = suf
# verdicts
out["verdict"] = {}
for k in ks:
    rn = [nec[f"rand{k}_{r}"] for r in range(a.n_random)]; rs = [suf[f"rand{k}_{r}"] for r in range(a.n_random)]
    out["verdict"][f"k{k}"] = {"necessity_top": round(nec[f"top{k}"], 4), "necessity_random": [round(x, 4) for x in rn],
                               "necessity_beats_all_random": nec[f"top{k}"] > max(rn),
                               "sufficiency_top": round(suf[f"top{k}"], 4), "sufficiency_random": [round(x, 4) for x in rs],
                               "sufficiency_beats_all_random": suf[f"top{k}"] > max(rs)}
out["G3a_pass_k40"] = out["verdict"].get("k40", {}).get("necessity_beats_all_random")
out["seconds"] = round(time.time() - t0); out["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
os.makedirs(a.out_dir, exist_ok=True)
json.dump(out, open(f"{a.out_dir}/validation_{a.lang}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k not in ("necessity_drop", "sufficiency_recovery")}, indent=1, ensure_ascii=False))
