"""J.1 — leak check between each fine-tuning set and every evaluation set: exact, MinHash (CuratorKIT's
3-gram/128-perm MinHash) and embedding (multilingual sentence-transformer) similarity.

    python scripts/data/leak_check.py v1      # before training: overlaps are dropped from the FINE-TUNING sets
    python scripts/data/leak_check.py v2      # after ClinicSafe-Hi: overlaps are dropped from the TEST set and noted

Writes Datasets/final/leak_report.json (and the v1/v2 copies)."""
import json, re, sys, unicodedata
from pathlib import Path
import numpy as np
from common import *
from curatorkit.normalizers.dedup import _ngrams, _make_hash_funcs, _minhash_signature, _jaccard_estimate

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1"
MINHASH_T, EMB_T = 0.85, 0.92          # CuratorKIT defaults (minhash_threshold 0.85, embedding_threshold 0.92)
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def norm(t):
    t = unicodedata.normalize("NFKC", t).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()

def sft_texts(path):
    rows = read_jsonl(path)
    return [(r["id"], r["messages"][0]["content"]) for r in rows]      # the user turn (prompt) is what could leak

def eval_sets():
    out = {}
    for d in sorted(FINAL.glob("*/prompts.jsonl")):
        name = d.parent.name
        if name.startswith("sft_"):
            continue
        out[name] = [(r["id"], r["text"]) for r in read_jsonl(d)]
    for d in sorted(FINAL.glob("*/calib_*.jsonl")):
        out[d.stem] = [(r["id"], r["text"]) for r in read_jsonl(d)]
    return out

if VERSION == "v1":      # before training: the formatted candidates
    ft = {"sft_medical": sft_texts(INTERIM / "medmcqa_sft_formatted.jsonl"),
          "sft_control": sft_texts(INTERIM / "alpaca_sft_formatted.jsonl")}
else:                    # after curation: the exact train sets the trainer reads
    ft = {"sft_medical": sft_texts(FINAL / "sft_medical" / "train.jsonl"),
          "sft_control": sft_texts(FINAL / "sft_control" / "train.jsonl")}
ev = eval_sets()
print("fine-tuning sets:", {k: len(v) for k, v in ft.items()})
print("evaluation sets:", {k: len(v) for k, v in ev.items()})

hf = _make_hash_funcs(128, 42)
def sig(t):
    g = _ngrams(norm(t), 3)
    return _minhash_signature(g, hf) if g else None

from sentence_transformers import SentenceTransformer
emb = SentenceTransformer(EMB_MODEL, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
def embed(texts):
    return emb.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

report = {"version": VERSION, "thresholds": {"minhash_jaccard": MINHASH_T, "embedding_cosine": EMB_T},
          "embedding_model": EMB_MODEL, "pairs": [], "summary": {}}
ev_sigs = {k: [(i, sig(t)) for i, t in v] for k, v in ev.items()}
ev_norm = {k: {norm(t): i for i, t in v} for k, v in ev.items()}
ev_emb = {k: embed([t for _, t in v]) for k, v in ev.items()}
for fk, frows in ft.items():
    fsig = [(i, sig(t)) for i, t in frows]
    fnorm = [(i, norm(t)) for i, t in frows]
    femb = embed([t for _, t in frows])
    for ek in ev:
        hits = []
        # exact
        for i, n in fnorm:
            if n in ev_norm[ek]:
                hits.append({"ft_id": i, "eval_id": ev_norm[ek][n], "method": "exact", "score": 1.0})
        # minhash (all pairs; sets are small enough: 5k x <=1k)
        for i, s in fsig:
            if s is None: continue
            for j, s2 in ev_sigs[ek]:
                if s2 is None: continue
                jac = _jaccard_estimate(s, s2)
                if jac >= MINHASH_T:
                    hits.append({"ft_id": i, "eval_id": j, "method": "minhash", "score": round(jac, 3)})
        # embedding
        sims = femb @ ev_emb[ek].T
        for a, b in zip(*np.where(sims >= EMB_T)):
            hits.append({"ft_id": frows[a][0], "eval_id": ev[ek][b][0], "method": "embedding",
                         "score": round(float(sims[a, b]), 3)})
        report["summary"][f"{fk} x {ek}"] = {"exact": sum(h["method"] == "exact" for h in hits),
                                            "minhash": sum(h["method"] == "minhash" for h in hits),
                                            "embedding": sum(h["method"] == "embedding" for h in hits)}
        for h in hits:
            h.update({"ft_set": fk, "eval_set": ek}); report["pairs"].append(h)
        print(fk, "x", ek, report["summary"][f"{fk} x {ek}"])

# ---- action ----
flagged = {}
for h in report["pairs"]:
    flagged.setdefault(h["ft_set"], set()).add(h["ft_id"])
if VERSION == "v1":
    # drop overlapping rows from the fine-tuning sets (before training)
    for fk, src in (("sft_medical", "medmcqa_sft_formatted.jsonl"), ("sft_control", "alpaca_sft_formatted.jsonl")):
        rows = read_jsonl(INTERIM / src)
        keep = [r for r in rows if r["id"] not in flagged.get(fk, set())]
        write_jsonl(INTERIM / src.replace("_formatted", "_leakfree"), keep)
        report["summary"][f"{fk}_dropped"] = len(rows) - len(keep)
        print(fk, "dropped", len(rows) - len(keep), "->", len(keep))
else:
    # after training: drop the overlapping items from the TEST sets instead, and note them
    drop_eval = {}
    for h in report["pairs"]:
        drop_eval.setdefault(h["eval_set"], set()).add(h["eval_id"])
    for ek, ids in drop_eval.items():
        p = FINAL / ek / "prompts.jsonl"
        if not p.exists(): continue
        rows = read_jsonl(p)
        keep = [r for r in rows if r["id"] not in ids]
        write_jsonl(p, keep)
        report["summary"][f"{ek}_dropped_from_test"] = sorted(ids)
        print(ek, "dropped from test:", sorted(ids))
report["clean"] = len(report["pairs"]) == 0
json.dump(report, open(FINAL / f"leak_report_{VERSION}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(report, open(FINAL / "leak_report.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("overlaps found:", len(report["pairs"]), "| clean:", report["clean"])
