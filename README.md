# Sushovit (Task 1: setup, models, data)

**New machine? Follow [Startup.md](Startup.md).** Models and fine-tuned weights are not in Git (adapters shared via Google Drive); the Lexsi kits are re-cloned at pinned commits.

Lexsi AI Safety Hackathon · Track 2 · Team Manish & Sushovit. Planning docs (`Tasks/`) and the status report (`Report/`) are kept outside Git.

## Layout

```
.env                      HF tokens (git-ignored)           DECISIONS.md   every decision, dated
ENV.md / ENV-local.txt    pinned versions                   PREREG.md      pre-registration draft
requirements.txt          pinned Python packages            KITS/          kit clones at pinned commits (git-ignored, see Startup.md)
Datasets/raw              downloads as fetched (XSTest CSV)  Datasets/interim  samples, sheets, drafts, stats
Datasets/final/<set>/prompts.jsonl + manifest.json + checksums.txt         the team-schema sets
scripts/download_all.py   Task 1 F                          scripts/smoke/g*.py    Task 1 G smoke tests
scripts/data/*.py         Task 1 I/J curation               scripts/eval/medmcqa_score.py  one-token MedMCQA accuracy
finetune/train_lora.py, merge.py  Phase 3 LoRA + merge     locate/   Phase 5 CircuitKIT runs + tensor map   colab/                 Colab first-cell template (free T4 only)
models/                   hf_cache (HF_HOME), runs (adapters), merged (full weights)
results/smoke/*.json      smoke-test numbers (peak VRAM)     logs/                  run logs
```

Environment: Python 3.12 venv (`.venv/`, see Startup.md); **all model files and caches live in `models/`** (`models/hf_cache` = `HF_HOME`, `models/runs` = LoRA runs, `models/merged` = merged weights); `PYTHONUTF8=1`. Nothing is pushed to any remote (local-only rule, 3 Nov).

## Reproduce the data package (seed 13 everywhere)

```powershell
python scripts/download_all.py
cd scripts/data
python build_medmcqa.py 5000        # I.1 SFT medical + I.3 MedMCQA test 500 / dev 300
python build_alpaca.py              # I.2 token-matched control set + Alpaca calibration pool
python build_aya.py                 # I.4 Aya EN/HI test 200 + dev 50 + calibration pools; Hinglish sheet
python build_xstest.py              # I.6.1 XSTest EN test 150 / dev 100; Hindi sheet
python translate_indictrans2.py xstest ; python translate_indictrans2.py calib   # Hindi drafts
# manual: Hinglish rewrites (Datasets/interim/hinglish), XSTest post-edit (xstest_en150_sheet.jsonl),
#         ClinicSafe-Hi (Datasets/interim/clinicsafe_hi/*.jsonl), calibration Hinglish (Datasets/interim/calibration)
python assemble_manual.py hinglish ; python assemble_manual.py clinicsafe ; python assemble_manual.py xstest
python build_calibration.py sheets ; python build_calibration.py final
python leak_check.py v1             # drops overlaps from the fine-tuning sets (before training)
python curate_sft.py medical ; python curate_sft.py control        # CuratorKIT: clean, MinHash dedup, toxicity gate, export
python qwen3guard_check.py          # ClinicSafe-Hi prompt-moderation check
python leak_check.py v2             # after ClinicSafe-Hi: overlaps dropped from the test set and noted
python package.py build ; python package.py push
```

Schema of every evaluation record: `id, lang (en|hi|hinglish), split (test|dev|calib), set, source, category, twin_id, text, gloss_en, origin, source_id`.
