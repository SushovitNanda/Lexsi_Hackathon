"""Phase 5 prep — contrast pairs per language for CircuitKIT (harmful prompt vs harmless prompt), built from dev and
calibration items only, never test items (DECISIONS.md). Prompts are stored raw; the runner renders them WITHOUT the
System Preamble (template off) because the preamble-length prompts do not fit the 12 GB card (G.5).

Pairs are matched by token length (|len_h − len_b| ≤ 2 tokens, no template) so EAP's position alignment holds."""
import json
from collections import defaultdict
from common import *

tok = tokenizer()
def ntok(t): return len(tok(t, add_special_tokens=False).input_ids)

harm = defaultdict(list); safe = defaultdict(list)
for r in read_jsonl(FINAL / "calibration" / "calib_trilingual_harmful.jsonl"): harm[r["lang"]].append(r)
for r in read_jsonl(FINAL / "calibration" / "calib_trilingual_harmless.jsonl"): safe[r["lang"]].append(r)
for r in read_jsonl(FINAL / "calibration" / "calib_en_only_harmful.jsonl"): harm["en"].append(r)
for r in read_jsonl(FINAL / "calibration" / "calib_en_only_harmless.jsonl"): safe["en"].append(r)
for lang in LANGS:                                         # harmful dev items too (dev is allowed for the circuit map)
    harm[lang] += read_jsonl(FINAL / f"aya_{lang}_dev" / "prompts.jsonl")
safe["en"] += read_jsonl(FINAL / "xstest_en_dev" / "prompts.jsonl")   # benign dev (English only)

out_dir = FINAL / "circuit_pairs"; out_dir.mkdir(exist_ok=True)
summary = {}
for lang in LANGS:
    H = [(r, ntok(r["text"])) for r in harm[lang]]
    S = [(r, ntok(r["text"])) for r in safe[lang]]
    used, pairs = set(), []
    r_ = rng(SEED + 40); r_.shuffle(H)
    for h, nh in H:
        best = None
        for i, (s, ns) in enumerate(S):
            if i in used: continue
            d = abs(nh - ns)
            if d <= 2 and (best is None or d < best[1]):
                best = (i, d)
                if d == 0: break
        if best is None: continue
        used.add(best[0]); s = S[best[0]][0]
        pairs.append({"pair_id": f"cp-{lang}-{len(pairs):04d}", "lang": lang, "harmful_id": h["id"], "harmful": h["text"],
                      "benign_id": s["id"], "benign": s["text"], "n_tok_harmful": nh, "n_tok_benign": S[best[0]][1],
                      "harmful_split": h["split"], "benign_split": s["split"]})
    write_jsonl(out_dir / f"{lang}.jsonl", pairs)
    summary[lang] = {"harmful_pool": len(H), "harmless_pool": len(S), "pairs": len(pairs)}
    print(lang, summary[lang])
json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=1)
