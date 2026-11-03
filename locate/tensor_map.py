"""Phase 5 outputs from the CircuitKIT score side-cars (Sushovit_Task §7.4, RQ8):
  * per run: the 36 × 7 layer × matrix score table in GGUF tensor names (attn_q/k/v/output from the layer's heads,
    ffn_gate/up/down from the layer's MLP) + CircuitKIT's layer_suggestions / safety_units passthrough
  * across runs: Jaccard overlap of the top-k nodes (6 × 6), and the "what changed" ranking (orig − med per node)

    python locate/tensor_map.py            # reads results/circuits/*_scores.json, writes results/circuits/tables/
"""
import glob, json, os, re
from collections import defaultdict

OUT = "results/circuits/tables"; os.makedirs(OUT, exist_ok=True)
N_LAYERS, TOPK = 36, 60
ATTN = ["attn_q", "attn_k", "attn_v", "attn_output"]; FFN = ["ffn_gate", "ffn_up", "ffn_down"]

def parse(name):
    m = re.match(r"A(\d+)\.(\d+)$", name)
    if m: return "attn", int(m.group(1)), int(m.group(2))
    m = re.match(r"MLP (\d+)$", name)
    if m: return "mlp", int(m.group(1)), None
    return None, None, None

runs = {}
for p in sorted(glob.glob("results/circuits/*_scores.json")):
    tag = os.path.basename(p).replace("_scores.json", "")
    s = json.load(open(p, encoding="utf-8"))
    runs[tag] = s
    table = {l: {k: 0.0 for k in ATTN + FFN} for l in range(N_LAYERS)}
    for node, sc in s["node_scores"].items():
        kind, layer, head = parse(node)
        if kind == "attn":
            for k in ATTN: table[layer][k] += abs(sc)
        elif kind == "mlp":
            for k in FFN: table[layer][k] += abs(sc)
    rows = [{"layer": l, **{k: round(v, 5) for k, v in table[l].items()}} for l in range(N_LAYERS)]
    json.dump({"run": tag, "model": s.get("model"), "task": s.get("task"), "algorithm": s.get("algorithm"),
               "layer_x_matrix": rows, "layer_suggestions": s.get("layer_suggestions"), "safety_units": s.get("safety_units"),
               "top_nodes": sorted(s["node_scores"].items(), key=lambda x: -abs(x[1]))[:TOPK]},
              open(f"{OUT}/{tag}_layer_x_matrix.json", "w", encoding="utf-8"), indent=1)
    with open(f"{OUT}/{tag}_layer_x_matrix.csv", "w", encoding="utf-8") as f:
        f.write("layer," + ",".join(ATTN + FFN) + "\n")
        for r in rows: f.write(str(r["layer"]) + "," + ",".join(str(r[k]) for k in ATTN + FFN) + "\n")
    top_layers = sorted(range(N_LAYERS), key=lambda l: -sum(table[l].values()))[:6]
    print(f"{tag}: top layers by total score {top_layers}; top nodes {[n for n, _ in sorted(s['node_scores'].items(), key=lambda x: -abs(x[1]))[:6]]}")

# overlap matrix (RQ8) on top-k nodes
tags = list(runs); tops = {t: set(n for n, _ in sorted(runs[t]["node_scores"].items(), key=lambda x: -abs(x[1]))[:TOPK]) for t in tags}
jac = {a: {b: round(len(tops[a] & tops[b]) / len(tops[a] | tops[b]), 3) for b in tags} for a in tags}
json.dump({"top_k": TOPK, "jaccard": jac}, open(f"{OUT}/overlap_jaccard.json", "w", encoding="utf-8"), indent=1)
print("Jaccard (top-%d):" % TOPK)
for a in tags: print(" ", a, {b: jac[a][b] for b in tags if b != a})

# what changed: orig − med per node, per language
for lang in ("en", "hi", "hinglish"):
    o = [t for t in tags if t.startswith("orig_" + lang + "_")]; m = [t for t in tags if t.startswith("med_" + lang + "_")]
    if o and m:
        so, sm = runs[o[0]]["node_scores"], runs[m[0]]["node_scores"]
        diff = sorted(((n, abs(so[n]) - abs(sm.get(n, 0.0)), so[n], sm.get(n)) for n in so), key=lambda x: -x[1])
        json.dump([{"node": n, "drop": round(d, 5), "orig": round(a, 5), "med": round(b, 5) if b is not None else None} for n, d, a, b in diff],
                  open(f"{OUT}/what_changed_{lang}.json", "w", encoding="utf-8"), indent=1)
        print(lang, "largest importance drops orig→med:", [(n, round(d, 3)) for n, d, _, _ in diff[:8]])
