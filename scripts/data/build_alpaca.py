"""I.2 — control SFT set from Alpaca-cleaned, token-matched (within ~2%) to the medical set at a similar row count.
Also reserves a pool of Alpaca prompts NOT in the control set for the steering calibration pairs (I.9)."""
import glob, json
from common import *

stats = json.load(open(INTERIM / "medmcqa_sft_stats.json", encoding="utf-8"))
target_tokens, target_rows = stats["tokens"], stats["rows"]
src = f"{hf_snapshot('yahma/alpaca-cleaned')}/alpaca_data_cleaned.json"
data = json.load(open(src, encoding="utf-8"))
print("alpaca-cleaned rows:", len(data))
r = rng(SEED)
idx = list(range(len(data))); r.shuffle(idx)

def to_msgs(ex):
    user = ex["instruction"].strip()
    if ex.get("input", "").strip():
        user += "\n\n" + ex["input"].strip()
    return [{"role": "user", "content": user}, {"role": "assistant", "content": ex["output"].strip()}]

mean_target = target_tokens / target_rows
rows, tot = [], 0
lo, hi = 0.6 * mean_target, 1.6 * mean_target     # keep rows near the medical mean so rows and tokens both match
reserve = []                                       # unused Alpaca prompts for calibration pairs
for i in idx:
    ex = data[i]
    if not ex["output"].strip() or len(ex["output"]) < 40:
        continue
    msgs = to_msgs(ex)
    n = chat_tokens(msgs)
    if n > MAX_TOKENS:
        continue
    if tot >= target_tokens * 0.995 or len(rows) >= target_rows * 1.05:
        reserve.append({"id": f"alpaca-{i}", "source_id": i, "text": msgs[0]["content"], "n_tokens": n})
        if len(reserve) >= 2000:
            break
        continue
    if not (lo <= n <= hi):
        continue
    rows.append({"id": f"alpaca-train-{i}", "source_id": i, "n_tokens": n, "messages": msgs})
    tot += n
print(f"control SFT: {len(rows)} rows, {tot} tokens (medical {target_rows} rows, {target_tokens} tokens; "
      f"ratio {tot/target_tokens:.4f})")
assert abs(tot / target_tokens - 1) <= 0.02, "token match outside 2%"
write_jsonl(INTERIM / "alpaca_sft_formatted.jsonl", rows)
write_jsonl(INTERIM / "alpaca_calibration_pool.jsonl", reserve)
json.dump({"rows": len(rows), "tokens": tot, "medical_rows": target_rows, "medical_tokens": target_tokens,
           "ratio": tot / target_tokens, "seed": SEED, "calibration_pool": len(reserve)},
          open(INTERIM / "alpaca_sft_stats.json", "w", encoding="utf-8"), indent=1)
