"""I.9 — steering calibration pairs, all outside every test set.

    python scripts/data/build_calibration.py sheets     # draws the items; writes the manual sheets
    python scripts/data/build_calibration.py final      # assembles the final calib_* files after the manual steps

English-only (RQ9): 100 Aya EN items from the calibration pool vs 100 Alpaca prompts not in the control set.
Three-language: 33 EN + 33 HI Aya items from the pools + 33 Hinglish rewrites of 33 further HI pool items (manual);
harmless side: 33 Alpaca EN prompts, 33 translated to Hindi (IndicTrans2 + post-edit), 33 rewritten in Hinglish.
"""
import json, re, sys
from common import *

job = sys.argv[1]
CAL = INTERIM / "calibration"; CAL.mkdir(exist_ok=True)

if job == "sheets":
    en_pool = read_jsonl(INTERIM / "aya_en_calibration_pool.jsonl")
    hi_pool = read_jsonl(INTERIM / "aya_hi_calibration_pool.jsonl")
    al_pool = read_jsonl(INTERIM / "alpaca_calibration_pool.jsonl")
    # harmless prompts: short, instruction-like, no input blocks, 6-40 words
    HARM = re.compile(r"\b(misrepresent|fake|lie|lies|steal|hack|kill|weapon|gun|bomb|drug|violen|racis|sexis|insult|"
                      r"offensive|manipulat|scam|fraud|cheat|illegal|harm|attack|murder|suicide|porn|sexual|hate)\w*", re.I)
    BAD = re.compile(r"\b(below|following|given|above|this (text|sentence|paragraph|list|code|article|passage|essay|data))\b", re.I)
    al_pool = [r for r in al_pool if 6 <= len(r["text"].split()) <= 40 and "\n" not in r["text"]
               and not r["text"].rstrip().endswith(":") and not BAD.search(r["text"])
               and not HARM.search(r["text"])]   # the harmless side must be unambiguously harmless
    en100, en_rest = stratified_sample(en_pool, lambda r: r["category"].split("|")[0], 100, seed=SEED + 10)
    en33, _ = stratified_sample(en_rest, lambda r: r["category"].split("|")[0], 33, seed=SEED + 11)
    hi33, hi_rest = stratified_sample(hi_pool, lambda r: r["category"].split("|")[0], 33, seed=SEED + 12)
    hg33_src, _ = stratified_sample(hi_rest, lambda r: r["category"].split("|")[0], 33, seed=SEED + 13)
    r = rng(SEED + 14); r.shuffle(al_pool)
    al100, al33_en, al33_hi, al33_hg = al_pool[:100], al_pool[100:133], al_pool[133:166], al_pool[166:199]
    write_jsonl(CAL / "en_only_harmful.jsonl", en100)
    write_jsonl(CAL / "en_only_harmless.jsonl", al100)
    write_jsonl(CAL / "tri_harmful_en.jsonl", en33)
    write_jsonl(CAL / "tri_harmful_hi.jsonl", hi33)
    write_jsonl(CAL / "tri_harmful_hg_sheet.jsonl",
                [{"id": x["id"], "hindi": x["text"], "gloss_en": x["gloss_en"], "category": x["category"],
                  "split": "calib", "source_id": x["source_id"], "hinglish": ""} for x in hg33_src])
    write_jsonl(CAL / "tri_harmless_en.jsonl", al33_en)
    write_jsonl(INTERIM / "calib_harmless_en_for_hi.jsonl", al33_hi)          # → translate_indictrans2.py calib
    write_jsonl(CAL / "tri_harmless_hg_sheet.jsonl",
                [{"id": x["id"], "english": x["text"], "source_id": x["source_id"], "hinglish": ""} for x in al33_hg])
    print({k: len(v) for k, v in dict(en100=en100, al100=al100, en33=en33, hi33=hi33, hg33=hg33_src,
                                        al33_en=al33_en, al33_hi=al33_hi, al33_hg=al33_hg).items()})

elif job == "final":
    out = FINAL / "calibration"; out.mkdir(exist_ok=True)
    def rec(x, i, lang, set_, src, text, **kw):
        return record(f"calib-{src}-{lang}-{i:04d}", lang, "calib", set_, src, text, **kw)
    # English-only
    harm = [rec(x, i, "en", "harmful_general", "aya_redteaming", x["text"], category=x["category"],
                source_id=x["source_id"]) for i, x in enumerate(read_jsonl(CAL / "en_only_harmful.jsonl"))]
    safe = [rec(x, i, "en", "benign_general", "alpaca", x["text"], source_id=str(x["source_id"]))
            for i, x in enumerate(read_jsonl(CAL / "en_only_harmless.jsonl"))]
    write_jsonl(out / "calib_en_only_harmful.jsonl", harm); write_jsonl(out / "calib_en_only_harmless.jsonl", safe)
    # three-language
    harm3 = [rec(x, i, "en", "harmful_general", "aya_redteaming", x["text"], category=x["category"],
                 source_id=x["source_id"]) for i, x in enumerate(read_jsonl(CAL / "tri_harmful_en.jsonl"))]
    harm3 += [rec(x, i, "hi", "harmful_general", "aya_redteaming", x["text"], category=x["category"],
                  gloss_en=x["gloss_en"], source_id=x["source_id"]) for i, x in enumerate(read_jsonl(CAL / "tri_harmful_hi.jsonl"))]
    hg = read_jsonl(CAL / "tri_harmful_hg_out.jsonl")
    harm3 += [rec(x, i, "hinglish", "harmful_general", "aya_redteaming", x["hinglish"].strip(), category=x["category"],
                  gloss_en=x["gloss_en"], origin="rewritten", source_id=x["source_id"]) for i, x in enumerate(hg)]
    safe3 = [rec(x, i, "en", "benign_general", "alpaca", x["text"], source_id=str(x["source_id"]))
             for i, x in enumerate(read_jsonl(CAL / "tri_harmless_en.jsonl"))]
    hi = read_jsonl(CAL / "tri_harmless_hi_out.jsonl")
    safe3 += [rec(x, i, "hi", "benign_general", "alpaca", x["hindi"].strip(), gloss_en=x["text"],
                  origin="indictrans2_postedited", source_id=str(x["source_id"])) for i, x in enumerate(hi)]
    hgs = read_jsonl(CAL / "tri_harmless_hg_out.jsonl")
    safe3 += [rec(x, i, "hinglish", "benign_general", "alpaca", x["hinglish"].strip(), gloss_en=x["english"],
                  origin="rewritten", source_id=str(x["source_id"])) for i, x in enumerate(hgs)]
    write_jsonl(out / "calib_trilingual_harmful.jsonl", harm3); write_jsonl(out / "calib_trilingual_harmless.jsonl", safe3)
    print({"en_only": (len(harm), len(safe)), "trilingual": (len(harm3), len(safe3))})
    # no calibration id may equal a test-set id or text
    test_texts = set()
    for p in FINAL.glob("*_test/prompts.jsonl"):
        test_texts |= {r["text"].strip().lower() for r in read_jsonl(p)}
    for p in FINAL.glob("clinicsafe_hi_*/prompts.jsonl"):
        test_texts |= {r["text"].strip().lower() for r in read_jsonl(p)}
    leak = [r["id"] for r in harm + safe + harm3 + safe3 if r["text"].strip().lower() in test_texts]
    print("calibration items overlapping a test set:", leak)
    assert not leak
