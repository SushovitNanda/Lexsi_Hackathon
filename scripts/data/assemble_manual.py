"""Turn the manual sheets into final sets in the team schema.

    python scripts/data/assemble_manual.py hinglish     # I.5  → aya_hinglish_test (150) + aya_hinglish_dev (50)
    python scripts/data/assemble_manual.py clinicsafe   # I.7  → clinicsafe_hi_{en,hi,hinglish}
    python scripts/data/assemble_manual.py xstest       # I.6  → xstest_hi_test, xstest_hinglish_test (150 each)
"""
import json, re, sys
from collections import Counter
from common import *

job = sys.argv[1]
DEVA = re.compile(r"[ऀ-ॿ]")

if job == "hinglish":
    rows = []
    for k in range(4):
        rows += read_jsonl(INTERIM / "hinglish" / f"out_{k:02d}.jsonl")
    assert len(rows) == 200
    test, dev = [], []
    for r in rows:
        assert r["hinglish"].strip() and not DEVA.search(r["hinglish"]), r["id"]
        split = r["split"]
        i = len(test) if split == "test" else len(dev)
        rec = record(f"aya-hinglish-{split}-{i:04d}", "hinglish", split, "harmful_general", "aya_redteaming",
                     r["hinglish"].strip(), category=r["category"], gloss_en=r["gloss_en"], origin="rewritten",
                     source_id=r["source_id"], hindi_id=r["id"], reviewed_by="")
        (test if split == "test" else dev).append(rec)
    assert len(test) == 150 and len(dev) == 50, (len(test), len(dev))
    write_jsonl(FINAL / "aya_hinglish_test" / "prompts.jsonl", test)
    write_jsonl(FINAL / "aya_hinglish_dev" / "prompts.jsonl", dev)
    print("hinglish test/dev:", len(test), len(dev), Counter(r["category"].split("|")[0] for r in test))

elif job == "clinicsafe":
    allrows = []
    for code in ("rx", "doc", "self", "misinfo", "priv", "scam"):
        allrows += read_jsonl(INTERIM / "clinicsafe_hi" / f"{code}.jsonl")
    ids = {r["id"] for r in allrows}
    assert len(ids) == len(allrows), "duplicate ids"
    for r in allrows:
        assert r["twin_id"] in ids, r["id"]
        assert r["lang"] in LANGS and r["set"] in ("harmful_clinical", "benign_clinical")
        if r["lang"] == "hi":
            assert DEVA.search(r["text"]), r["id"]
        else:
            assert not DEVA.search(r["text"]), r["id"]
        r.setdefault("origin", "original" if r["lang"] == "en" else "adapted")
        r["split"] = "test"; r["source"] = "clinicsafe_hi"
    for lang in LANGS:
        sub = [r for r in allrows if r["lang"] == lang]
        write_jsonl(FINAL / f"clinicsafe_hi_{lang}" / "prompts.jsonl", sub)
        print(lang, len(sub), Counter((r["set"], r["category"]) for r in sub).most_common(3), "...")
    print("total", len(allrows))

elif job == "xstest":
    sheet = read_jsonl(INTERIM / "xstest_en150_sheet.jsonl")
    hi, hg = [], []
    for i, r in enumerate(sheet):
        assert r["hindi"].strip() and DEVA.search(r["hindi"]), r["id"]
        assert r["hinglish"].strip() and not DEVA.search(r["hinglish"]), r["id"]
        assert r["origin"] in ("indictrans2_postedited", "replaced", "adapted"), (r["id"], r["origin"])
        base = dict(category=r["type"], gloss_en=r.get("gloss_en") or r["english"], source_id=r["source_id"],
                    english_id=r["id"], replaced_english=r.get("replaced_english"))
        hi.append(record(f"xstest-hi-test-{i:04d}", "hi", "test", "benign_general", "xstest", r["hindi"].strip(),
                         origin=r["origin"], **base))
        hg.append(record(f"xstest-hinglish-test-{i:04d}", "hinglish", "test", "benign_general", "xstest",
                         r["hinglish"].strip(), origin="rewritten" if r["origin"] != "replaced" else "replaced", **base))
    write_jsonl(FINAL / "xstest_hi_test" / "prompts.jsonl", hi)
    write_jsonl(FINAL / "xstest_hinglish_test" / "prompts.jsonl", hg)
    print("xstest hi/hinglish:", len(hi), len(hg), Counter(r["origin"] for r in hi))
