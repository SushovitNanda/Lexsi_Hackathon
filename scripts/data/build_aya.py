"""I.4 — Aya Red-teaming English and Hindi: 200 test + 50 dev per language, stratified by (primary) harm
category; the remaining items form the steering calibration pool. Toxicity filter OFF (these must stay harmful).
Also writes the Hindi sheet (150 test + 50 dev) that the Hinglish rewrites (I.5) are made from."""
import glob, json
from common import *

snap = hf_snapshot("CohereForAI/aya_redteaming")
files = {"en": f"{snap}/aya_eng.jsonl", "hi": f"{snap}/aya_hin.jsonl"}
summary = {}
for lang, path in files.items():
    raw = read_jsonl(path)
    items = []
    for i, x in enumerate(raw):
        cats = json.loads(x["harm_category"]) if isinstance(x["harm_category"], str) else x["harm_category"]
        items.append({"source_id": f"aya_{ 'eng' if lang=='en' else 'hin'}_{i:04d}", "text": x["prompt"].strip(),
                      "categories": cats, "primary": cats[0] if cats else "Unknown",
                      "global_or_local": x.get("global_or_local"),
                      "gloss_en": (x.get("semantic_translation") or x.get("literal_translation")) if lang == "hi" else None})
    # drop exact duplicates of the prompt text first
    seen, uniq = set(), []
    for it in items:
        k = it["text"].lower()
        if k in seen: continue
        seen.add(k); uniq.append(it)
    test, rest = stratified_sample(uniq, lambda r: r["primary"], 200, seed=SEED)
    dev, pool = stratified_sample(rest, lambda r: r["primary"], 50, seed=SEED + 1)

    def rec(it, split, i):
        return record(f"aya-{lang}-{split}-{i:04d}", lang, split, "harmful_general", "aya_redteaming", it["text"],
                      category="|".join(it["categories"]), gloss_en=it["gloss_en"], origin="original",
                      source_id=it["source_id"], global_or_local=it["global_or_local"])
    trecs = [rec(it, "test", i) for i, it in enumerate(test)]
    drecs = [rec(it, "dev", i) for i, it in enumerate(dev)]
    crecs = [rec(it, "calib", i) for i, it in enumerate(pool)]
    write_jsonl(FINAL / f"aya_{lang}_test" / "prompts.jsonl", trecs)
    write_jsonl(FINAL / f"aya_{lang}_dev" / "prompts.jsonl", drecs)
    write_jsonl(INTERIM / f"aya_{lang}_calibration_pool.jsonl", crecs)
    from collections import Counter
    summary[lang] = {"raw": len(raw), "unique": len(uniq), "test": len(trecs), "dev": len(drecs), "pool": len(crecs),
                     "test_primary_cats": dict(Counter(it["primary"] for it in test)),
                     "all_primary_cats": dict(Counter(it["primary"] for it in uniq))}
    print(lang, json.dumps(summary[lang], indent=1))

    if lang == "hi":
        # I.0 decision: Hinglish = rewrites of 150 of the 200 Hindi test items (stratified) + all 50 dev items
        sub150, _ = stratified_sample(trecs, lambda r: r["category"].split("|")[0], 150, seed=SEED + 2)
        sheet = [{"id": r["id"], "hindi": r["text"], "gloss_en": r["gloss_en"], "category": r["category"],
                  "split": r["split"], "source_id": r["source_id"], "hinglish": ""} for r in sub150 + drecs]
        write_jsonl(INTERIM / "hinglish" / "aya_hi_to_rewrite.jsonl", sheet)
        print("Hinglish rewrite sheet:", len(sheet), "items (150 test + 50 dev)")
json.dump(summary, open(INTERIM / "aya_sample_stats.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
