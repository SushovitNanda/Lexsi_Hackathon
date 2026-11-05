"""Refusal / compliance opener token sets per language (Tasks/reference_notes.md §6, rule A4), from the original
model's own FP16 dev-pass answers (official template, greedy): results/private/dev_pass__orig__fp16/responses.jsonl.

  harmful pool  = harmful_general + harmful_clinical answers (120 per language)
  harmless pool = benign_general + benign_clinical answers (50 per language)
  R_l = first content tokens opening >= 3 harmful answers with harmful:harmless >= 3:1
  C_l = first content tokens opening >= 3 harmless answers with harmless:harmful >= 3:1
  (newline / whitespace-only tokens skipped; the first token with content is used)

    python locate/token_sets_v2.py [--cell dev_pass__orig__fp16] -> locate/token_sets_v2.json
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
HARM = {"harmful_general", "harmful_clinical"}; BEN = {"benign_general", "benign_clinical"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", default="dev_pass__orig__fp16"); ap.add_argument("--min_count", type=int, default=3)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
    rows = [json.loads(l) for l in open(ROOT / "results" / "private" / a.cell / "responses.jsonl", encoding="utf-8") if l.strip()]
    cnt = defaultdict(lambda: {"harmful": Counter(), "harmless": Counter()})
    for r in rows:
        ids = tok(r["text"], add_special_tokens=False).input_ids
        first = None
        for t in ids:
            if tok.decode([t]).strip():
                first = t; break
        if first is None: continue
        kind = "harmful" if r["set"] in HARM else "harmless" if r["set"] in BEN else None
        if kind: cnt[r["lang"]][kind][first] += 1
    out = {}
    for lang, c in cnt.items():
        H, S = c["harmful"], c["harmless"]
        R = [t for t, n in H.items() if n >= a.min_count and n >= 3 * S.get(t, 0)]
        C = [t for t, n in S.items() if n >= a.min_count and n >= 3 * H.get(t, 0)]
        out[lang] = {"refusal_ids": R, "refusal_tokens": [tok.decode([t]) for t in R],
                     "compliance_ids": C, "compliance_tokens": [tok.decode([t]) for t in C],
                     "refusal_coverage": round(sum(H[t] for t in R) / max(1, sum(H.values())), 3),
                     "compliance_coverage": round(sum(S[t] for t in C) / max(1, sum(S.values())), 3),
                     "n_harmful": sum(H.values()), "n_harmless": sum(S.values()),
                     "harmful_top": {tok.decode([t]): n for t, n in H.most_common(10)},
                     "harmless_top": {tok.decode([t]): n for t, n in S.most_common(10)}}
        print(lang, json.dumps({k: v for k, v in out[lang].items() if k.endswith("tokens") or k.endswith("coverage")}, ensure_ascii=False))
    json.dump(out, open(ROOT / "locate" / "token_sets_v2.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
