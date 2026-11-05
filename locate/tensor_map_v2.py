"""Circuits v2 tables (PREREG §7; Tasks/reference_notes.md §6):
  * per run: 36 x 7 layer x matrix score table in GGUF tensor names (+ layer_suggestions / safety_units passthrough)
  * MLP 0 excluded from every top-k (tied-embedding attribution artefact)
  * overlap matrix (top-60 Jaccard) across the full runs, with split-half floors per model x language
  * what changed: orig - med per node, per language
  * union tensor list for circuit-restricted ReSta / protection: HF parameter names of the top 5 / 10 / 20 % tensors
    by summed |score| over the ORIGINAL model's three language runs (each run normalised to sum 1)

    python locate/tensor_map_v2.py [--in results/circuits/v2] [--out results/circuits/v2/tables]
"""
import argparse, glob, json, os, re
from collections import defaultdict

N_LAYERS, TOPK = 36, 60
ATTN = ["attn_q", "attn_k", "attn_v", "attn_output"]; FFN = ["ffn_gate", "ffn_up", "ffn_down"]
HF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
      "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
TAG = re.compile(r"^(?P<model>[a-z0-9]+)_(?P<lang>en|hi|hinglish)(?:_half-(?P<half>[ab]))?_(?P<alg>.+)$")


def parse(name):
    m = re.match(r"A(\d+)\.(\d+)$", name)
    if m: return "attn", int(m.group(1)), int(m.group(2))
    m = re.match(r"MLP (\d+)$", name)
    if m: return "mlp", int(m.group(1)), None
    return None, None, None


def top_nodes(scores, k=TOPK):
    return [n for n, _ in sorted(((n, s) for n, s in scores.items() if n != "MLP 0"), key=lambda x: -abs(x[1]))[:k]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default="results/circuits/v2"); ap.add_argument("--out", default="results/circuits/v2/tables")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    runs, tables = {}, {}
    for p in sorted(glob.glob(f"{a.inp}/*_scores.json")):
        tag = os.path.basename(p).replace("_scores.json", ""); m = TAG.match(tag)
        if not m: print("skip", tag); continue
        s = json.load(open(p, encoding="utf-8")); runs[tag] = {**m.groupdict(), "scores": s["node_scores"], "raw": s}
        table = {l: {k: 0.0 for k in ATTN + FFN} for l in range(N_LAYERS)}
        for node, sc in s["node_scores"].items():
            kind, layer, _ = parse(node)
            if kind == "attn":
                for k in ATTN: table[layer][k] += abs(sc)
            elif kind == "mlp":
                for k in FFN: table[layer][k] += abs(sc)
        tables[tag] = table
        rows = [{"layer": l, **{k: round(v, 5) for k, v in table[l].items()}} for l in range(N_LAYERS)]
        json.dump({"run": tag, **m.groupdict(), "layer_x_matrix": rows, "layer_suggestions": s.get("layer_suggestions"),
                   "safety_units": s.get("safety_units"), "top_nodes_excl_mlp0": top_nodes(s["node_scores"])},
                  open(f"{a.out}/{tag}_layer_x_matrix.json", "w", encoding="utf-8"), indent=1)
        with open(f"{a.out}/{tag}_layer_x_matrix.csv", "w", encoding="utf-8") as f:
            f.write("layer," + ",".join(ATTN + FFN) + "\n")
            for r in rows: f.write(str(r["layer"]) + "," + ",".join(str(r[k]) for k in ATTN + FFN) + "\n")
    full = [t for t, r in runs.items() if not r["half"]]
    tops = {t: set(top_nodes(runs[t]["scores"])) for t in runs}
    jac = lambda x, y: round(len(tops[x] & tops[y]) / max(1, len(tops[x] | tops[y])), 3)
    overlap = {x: {y: jac(x, y) for y in full} for x in full}
    floors = {}
    for t in full:
        ha, hb = t.replace(f"_{runs[t]['alg']}", f"_half-a_{runs[t]['alg']}"), t.replace(f"_{runs[t]['alg']}", f"_half-b_{runs[t]['alg']}")
        if ha in runs and hb in runs: floors[t] = jac(ha, hb)
    json.dump({"top_k": TOPK, "mlp0_excluded": True, "jaccard_full_runs": overlap, "split_half_floor": floors},
              open(f"{a.out}/overlap_jaccard.json", "w", encoding="utf-8"), indent=1)
    print("Jaccard (top-%d, MLP 0 excluded):" % TOPK)
    for x in full: print(" ", x, {y: overlap[x][y] for y in full if y != x}, "| split-half floor:", floors.get(x))
    for lang in ("en", "hi", "hinglish"):
        o = [t for t in full if runs[t]["model"] == "orig" and runs[t]["lang"] == lang]; m = [t for t in full if runs[t]["model"] == "med" and runs[t]["lang"] == lang]
        if o and m:
            so, sm = runs[o[0]]["scores"], runs[m[0]]["scores"]
            diff = sorted(((n, abs(so[n]) - abs(sm.get(n, 0.0)), so[n], sm.get(n)) for n in so if n != "MLP 0"), key=lambda x: -x[1])
            json.dump([{"node": n, "drop": round(d, 5), "orig": round(x, 5), "med": round(y, 5) if y is not None else None} for n, d, x, y in diff],
                      open(f"{a.out}/what_changed_{lang}.json", "w", encoding="utf-8"), indent=1)
            print(lang, "largest importance drops orig→med:", [(n, round(d, 3)) for n, d, _, _ in diff[:8]])
    # union tensor list from the original model's full runs
    orig_runs = [t for t in full if runs[t]["model"] == "orig"]
    if orig_runs:
        score = defaultdict(float)
        for t in orig_runs:
            tot = sum(v for l in tables[t].values() for k, v in l.items()) or 1.0
            for l in range(N_LAYERS):
                for k in ATTN + FFN:
                    if not (k in FFN and l == 0):            # MLP 0 excluded
                        score[(l, k)] += tables[t][l][k] / tot
        ranked = sorted(score.items(), key=lambda x: -x[1])
        n = len(ranked); lists = {}
        for pct in (5, 10, 20):
            k = max(1, round(n * pct / 100))
            lists[f"top{pct}"] = [f"model.layers.{l}.{HF[m]}.weight" for (l, m), _ in ranked[:k]]
            json.dump({"tensors": lists[f"top{pct}"], "pct": pct, "source_runs": orig_runs, "mlp0_excluded": True},
                      open(f"locate/cresta_tensors_v2_top{pct}.json", "w", encoding="utf-8"), indent=1)
        json.dump({"ranked": [{"layer": l, "matrix": m, "hf": f"model.layers.{l}.{HF[m]}.weight", "score": round(s, 6)} for (l, m), s in ranked],
                   "lists": lists, "source_runs": orig_runs}, open("locate/cresta_tensors_v2.json", "w", encoding="utf-8"), indent=1)
        print("tensor lists:", {k: len(v) for k, v in lists.items()}, "top 8:", [f"{l}.{m}" for (l, m), _ in ranked[:8]])


if __name__ == "__main__":
    main()
