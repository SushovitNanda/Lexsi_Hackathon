"""J.2 — package every final set (manifest, card, checksums) and push to the Hub.

    python scripts/data/package.py build            # manifests + checksums + cards locally
    python scripts/data/package.py push             # private fine-tuning dataset + gated clinicsafe-hi

Namespace: the HF user until the team org exists (DECISIONS.md)."""
import datetime, json, os, sys
from collections import Counter
from common import *

job = sys.argv[1]
NS = os.environ.get("HF_NAMESPACE", "SushiOrbit")
GATED = f"{NS}/clinicsafe-hi"
PRIVATE_SFT = f"{NS}/tinyaya-sft-private"

EVAL_SETS = ["aya_en_test", "aya_en_dev", "aya_hi_test", "aya_hi_dev", "aya_hinglish_test", "aya_hinglish_dev",
             "xstest_en_test", "xstest_en_dev", "xstest_hi_test", "xstest_hinglish_test",
             "clinicsafe_hi_en", "clinicsafe_hi_hi", "clinicsafe_hi_hinglish",
             "medmcqa_test_500", "medmcqa_dev_300", "calibration"]

def manifest_for(d):
    files = sorted(p for p in d.glob("*.jsonl"))
    rows = []
    for p in files:
        rows += read_jsonl(p)
    m = {"name": d.name, "created": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED,
         "files": {p.name: {"rows": len(read_jsonl(p)), "sha256": sha256(p)} for p in files},
         "rows": len(rows),
         "by_lang": dict(Counter(r["lang"] for r in rows)), "by_split": dict(Counter(r["split"] for r in rows)),
         "by_set": dict(Counter(r["set"] for r in rows)), "by_source": dict(Counter(r["source"] for r in rows)),
         "by_origin": dict(Counter(r.get("origin") for r in rows)),
         "by_category": dict(Counter(str(r.get("category")) for r in rows)),
         "schema": ["id", "lang", "split", "set", "source", "category", "twin_id", "text", "gloss_en", "origin", "source_id"],
         "contains_completions": False}
    json.dump(m, open(d / "manifest.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(d / "checksums.txt", "w", encoding="utf-8") as f:
        for p in files:
            f.write(f"{sha256(p)}  {p.name}\n")
    return m

if job == "build":
    all_m = {}
    for name in EVAL_SETS:
        d = FINAL / name
        if not d.exists():
            print("MISSING", name); continue
        all_m[name] = manifest_for(d); print(name, all_m[name]["rows"], all_m[name]["by_lang"])
    json.dump(all_m, open(FINAL / "manifest_all.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    # dataset card for the gated release
    card = open(ROOT / "scripts" / "data" / "clinicsafe_hi_card.md", encoding="utf-8").read()
    table = "\n".join(f"| `{k}` | {v['rows']} | {', '.join(f'{a}: {b}' for a, b in v['by_set'].items())} | {', '.join(f'{a}: {b}' for a, b in v['by_lang'].items())} |"
                      for k, v in all_m.items())
    card = card.replace("<<TABLE>>", "| set | rows | sets | languages |\n|---|---|---|---|\n" + table)
    (FINAL / "README.md").write_text(card, encoding="utf-8")
    print("card written")

elif job == "push":
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    # 1) gated evaluation dataset: prompts and labels only
    api.create_repo(GATED, repo_type="dataset", private=False, exist_ok=True)
    try:
        api.update_repo_settings(GATED, repo_type="dataset", gated="manual")
    except Exception as e:
        print("gating:", e)
    for name in EVAL_SETS:
        d = FINAL / name
        if d.exists():
            api.upload_folder(folder_path=str(d), path_in_repo=name, repo_id=GATED, repo_type="dataset",
                              allow_patterns=["*.jsonl", "manifest.json", "checksums.txt"])
            print("pushed", name)
    for f in ("README.md", "manifest_all.json", "leak_report.json"):
        if (FINAL / f).exists():
            api.upload_file(path_or_fileobj=str(FINAL / f), path_in_repo=f, repo_id=GATED, repo_type="dataset")
    print("gated dataset:", f"https://huggingface.co/datasets/{GATED}")
    # 2) private fine-tuning sets (CuratorKIT outputs)
    api.create_repo(PRIVATE_SFT, repo_type="dataset", private=True, exist_ok=True)
    for name in ("sft_medical", "sft_control"):
        d = FINAL / name
        if d.exists():
            api.upload_folder(folder_path=str(d), path_in_repo=name, repo_id=PRIVATE_SFT, repo_type="dataset",
                              ignore_patterns=["embedding_index/*"])
            print("pushed", name)
    if (FINAL / "leak_report.json").exists():
        api.upload_file(path_or_fileobj=str(FINAL / "leak_report.json"), path_in_repo="leak_report.json",
                        repo_id=PRIVATE_SFT, repo_type="dataset")
    print("private SFT dataset:", f"https://huggingface.co/datasets/{PRIVATE_SFT}")
