"""I.1 + I.3 — MedMCQA: medical SFT set (5,000 rows, stratified by subject, <=1,024 tokens with the official
template) and the held-out accuracy sets from the validation split (test 500, dev 300, disjoint)."""
import glob, re, json, sys
import pandas as pd
from common import *

SFT_N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
snap = f"{hf_snapshot('openlifescienceai/medmcqa')}/data"
train = pd.read_parquet(f"{snap}/train-00000-of-00001.parquet")
val = pd.read_parquet(f"{snap}/validation-00000-of-00001.parquet")
test_pub = pd.read_parquet(f"{snap}/test-00000-of-00001.parquet")
assert (test_pub.cop == -1).all(), "public test split should have no labels"

URL = re.compile(r"https?://\S+|www\.\S+")
MARKUP = re.compile(r"<[^>]+>")

def clean_exp(e):
    if not isinstance(e, str):
        return ""
    e = MARKUP.sub(" ", URL.sub(" ", e))
    e = re.sub(r"\s+", " ", e).strip()
    e = re.sub(r"^(ans(wer)?\s*[:.\-]?\s*)", "", e, flags=re.I)   # explanations often start with 'Ans. C ...'
    e = re.sub(r"^(is\s+)?['\"]?[a-d]['\"]?\s*(i\.e\.?|ie)?\s*[,.:\-]?\s*", "", e, flags=re.I)
    return e.strip()

def valid(df):
    df = df[(df.choice_type == "single") & df.cop.between(0, 3)].copy()
    df["exp_clean"] = df.exp.map(clean_exp)
    df = df[df.exp_clean.str.len() >= 60]            # non-trivial explanations only
    df = df[df.question.str.len() >= 15]
    for c in ("opa", "opb", "opc", "opd"):
        df = df[df[c].astype(str).str.strip().str.len() > 0]
    return df

# ---------- I.1 medical SFT set ----------
tr = valid(train)
print("train rows after filters:", len(tr))
# oversample x1.6 before token filtering so that 5,000 survive, then trim back stratified
pool, _ = stratified_sample(tr.to_dict("records"), lambda r: r["subject_name"], int(SFT_N * 1.6), seed=SEED)
rows, n_drop = [], 0
for q in pool:
    msgs = [{"role": "user", "content": medmcqa_user(q)},
            {"role": "assistant", "content": f"Answer: {LETTERS[q['cop']]}. {q['exp_clean']}"}]
    n = chat_tokens(msgs)
    if n > MAX_TOKENS:
        n_drop += 1; continue
    rows.append({"id": f"medmcqa-train-{q['id']}", "source_id": q["id"], "subject_name": q["subject_name"],
                 "topic_name": q.get("topic_name"), "n_tokens": n, "messages": msgs})
print(f"dropped {n_drop} over {MAX_TOKENS} tokens; {len(rows)} candidates")
rows, _ = stratified_sample(rows, lambda r: r["subject_name"], SFT_N, seed=SEED)
write_jsonl(INTERIM / "medmcqa_sft_formatted.jsonl", rows)
tot = sum(r["n_tokens"] for r in rows)
print(f"medical SFT: {len(rows)} rows, {tot} templated tokens, mean {tot/len(rows):.0f}")
json.dump({"rows": len(rows), "tokens": tot, "max_tokens": MAX_TOKENS, "seed": SEED,
           "subjects": pd.Series([r["subject_name"] for r in rows]).value_counts().to_dict()},
          open(INTERIM / "medmcqa_sft_stats.json", "w", encoding="utf-8"), indent=1)

# ---------- I.3 held-out accuracy sets (validation split) ----------
va = val[(val.choice_type == "single") & val.cop.between(0, 3)].copy()
va = va[va.question.str.len() >= 15]
for c in ("opa", "opb", "opc", "opd"):
    va = va[va[c].astype(str).str.strip().str.len() > 0]
print("validation rows after filters:", len(va))
vrows = va.to_dict("records")
test500, rest = stratified_sample(vrows, lambda r: r["subject_name"], 500, seed=SEED)
dev300, _ = stratified_sample(rest, lambda r: r["subject_name"], 300, seed=SEED + 1)
assert not ({r["id"] for r in test500} & {r["id"] for r in dev300})

def to_rec(q, split, i):
    return record(f"medmcqa-{split}-{i:04d}", "en", split, "medmcqa", "medmcqa", medmcqa_user(q),
                  category=q["subject_name"], source_id=q["id"], answer=LETTERS[q["cop"]],
                  options={"A": q["opa"], "B": q["opb"], "C": q["opc"], "D": q["opd"]})

write_jsonl(FINAL / "medmcqa_test_500" / "prompts.jsonl", [to_rec(q, "test", i) for i, q in enumerate(test500)])
write_jsonl(FINAL / "medmcqa_dev_300" / "prompts.jsonl", [to_rec(q, "dev", i) for i, q in enumerate(dev300)])
print("MedMCQA test 500 / dev 300 written; answer distribution test:",
      pd.Series([LETTERS[q["cop"]] for q in test500]).value_counts().to_dict())
