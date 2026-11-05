"""Circuits v2 contrast pairs per language (Tasks/reference_notes.md §6): dev and calibration items only, never test.

Harmful side : Aya dev (50/lang) + ClinicSafe dev twins pairs 021–030 harmful side + calibration harmful.
Benign side  : the twins of pairs 021–030 (true twins, paired first) + XSTest dev (EN/HI/Hinglish, 100 each)
               + calibration harmless, length-matched within 2 tokens (no template).
Writes Datasets/final/circuit_pairs_v2/{lang}.jsonl (+ summary.json). Alpaca-style calibration harmless items are
used last, so most benign sides are "alarming but safe" prompts.

    python scripts/data/build_circuit_pairs_v2.py
"""
import json, re
from collections import defaultdict
from common import *

tok = tokenizer()
def ntok(t): return len(tok(t, add_special_tokens=False).input_ids)
pat = re.compile(r"-(\d{3})(b?)$")
CS = {"en": "clinicsafe_dev_en", "hi": "clinicsafe_dev_hi", "hinglish": "clinicsafe_dev_hinglish"}

out_dir = FINAL / "circuit_pairs_v2"; out_dir.mkdir(exist_ok=True)
summary = {}
for lang in LANGS:
    pairs, used_b = [], set()
    # 1. true twins 021–030
    cs = {r["id"]: r for r in read_jsonl(FINAL / CS[lang] / "prompts.jsonl")}
    for i, r in cs.items():
        m = pat.search(i)
        if m.group(2) == "" and int(m.group(1)) > 20 and r["twin_id"] in cs:
            t = cs[r["twin_id"]]
            pairs.append({"pair_id": f"cp2-{lang}-{len(pairs):04d}", "lang": lang, "kind": "twin",
                          "harmful_id": i, "harmful": r["text"], "benign_id": t["id"], "benign": t["text"],
                          "n_tok_harmful": ntok(r["text"]), "n_tok_benign": ntok(t["text"])})
            used_b.add(t["id"])
    # 2. length-matched: Aya dev + calibration harmful  vs  XSTest dev (first), calibration harmless (last)
    H = read_jsonl(FINAL / f"aya_{lang}_dev" / "prompts.jsonl")
    H += [r for r in read_jsonl(FINAL / "calibration" / "calib_trilingual_harmful.jsonl") if r["lang"] == lang]
    S = read_jsonl(FINAL / f"xstest_{lang}_dev" / "prompts.jsonl")
    S_cal = [r for r in read_jsonl(FINAL / "calibration" / "calib_trilingual_harmless.jsonl") if r["lang"] == lang]
    Hn = [(r, ntok(r["text"])) for r in H]; r_ = rng(SEED + 41); r_.shuffle(Hn)
    for pool, kind in ((S, "xstest_dev"), (S_cal, "calibration")):
        Sn = [(r, ntok(r["text"])) for r in pool if r["id"] not in used_b]
        for h, nh in Hn:
            if any(p["harmful_id"] == h["id"] for p in pairs): continue
            best = None
            for j, (s, ns) in enumerate(Sn):
                if s["id"] in used_b: continue
                d = abs(nh - ns)
                if d <= 2 and (best is None or d < best[1]):
                    best = (j, d)
                    if d == 0: break
            if best is None: continue
            s, ns = Sn[best[0]]; used_b.add(s["id"])
            pairs.append({"pair_id": f"cp2-{lang}-{len(pairs):04d}", "lang": lang, "kind": kind,
                          "harmful_id": h["id"], "harmful": h["text"], "benign_id": s["id"], "benign": s["text"],
                          "n_tok_harmful": nh, "n_tok_benign": ns})
    write_jsonl(out_dir / f"{lang}.jsonl", pairs)
    summary[lang] = {"pairs": len(pairs), "by_kind": {k: sum(1 for p in pairs if p["kind"] == k) for k in ("twin", "xstest_dev", "calibration")},
                     "harmful_pool": len(H) + 10, "test_items": 0}
    print(lang, summary[lang])
json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=1)
