# Handoff to Manish — data package v1 (3 Nov 2026)

Written for: Manish (evaluation harness owner). Everything below is in `Datasets/final/` of this repo. Set the repo up with `Startup.md`; model weights, `results/` and `logs/` come over Google Drive.

## Part A — scripted sets (ready; load these first for the original-model cells)

| folder | rows | what |
|---|---|---|
| `aya_en_test`, `aya_en_dev` | 200, 50 | Aya Red-teaming English, stratified by primary harm category |
| `aya_hi_test`, `aya_hi_dev` | 200, 50 | Aya Red-teaming Hindi (`gloss_en` = Cohere's translation) |
| `xstest_en_test`, `xstest_en_dev` | 150, 100 | XSTest safe prompts, stratified by type (dev = benign dev set) |
| `medmcqa_test_500`, `medmcqa_dev_300` | 500, 300 | validation-split questions, one-token task; `answer` field holds the letter |

MedMCQA scoring: render the official template with the user turn in `text`, add the generation prompt, append `Answer:`,
and take the most likely of the four letter tokens ` A`/` B`/` C`/` D` (see `scripts/eval/medmcqa_score.py`).

## Part B — manual sets (drafted and self-checked; your half-review still due)

| folder | rows | what |
|---|---|---|
| `aya_hinglish_test`, `aya_hinglish_dev` | 150, 50 | Hinglish rewrites of 150 Hindi test + 50 Hindi dev items (`hindi_id`, same `source_id`) |
| `xstest_hi_test`, `xstest_hinglish_test` | 150, 150 | IndicTrans2 draft + post-edit; 12 replaced with clinical look-alikes (`origin`, `replaced_english`) |
| `clinicsafe_hi_en`, `clinicsafe_hi_hi`, `clinicsafe_hi_hinglish` | 204 each | 102 harmful + 102 benign twins per language, 6 categories, `twin_id` links pairs; `review` field carries the Qwen3Guard review note |
| `calibration/` | 398 | steering calibration pairs (`calib` split): EN-only 100/100 and trilingual 99/99 |
| `hand_labels/sample_to_label.jsonl` | 300 | the 100 prompts per language for hand labels (first 20 per language double-labelled) |

Per cell this gives 850 harmful (EN 200 + HI 200 + Hinglish 150 general; 102 clinical × 3 — 306, i.e. the plan's "~300") and
756 benign (XSTest 150 × 3 + 102 twins × 3), plus 500 MedMCQA.

## Schema

`id, lang (en|hi|hinglish), split (test|dev|calib), set (harmful_general|harmful_clinical|benign_general|benign_clinical|medmcqa),
source, category, twin_id, text, gloss_en, origin, source_id` (+ set-specific extras). UTF-8 JSONL, one record per line.
Each folder has `manifest.json` (counts by lang/split/set/origin, per-file sha256) and `checksums.txt`; the roll-up is
`Datasets/final/manifest_all.json`. Card: `Datasets/final/README.md`.

## Checks already done

- Leak check v1 (before training) and v2 (after all sets): exact + MinHash + multilingual-embedding dedup between the two
  fine-tuning sets and all 21 evaluation sets → **0 overlaps** in v2 (`Datasets/final/leak_report.json`).
- Qwen3Guard-Gen-4B prompt moderation on all 612 ClinicSafe-Hi items; 93 disagreements hand-reviewed (13 fixes, rest kept
  with reasons); re-check leaves 4 benign twins "Controversial" (kept, noted).
- Fine-tuning sets: CuratorKIT clean + MinHash + toxicity gate, token-matched (medical 4,815 rows / 2.650 M tokens; control
  5,004 rows / 2.676 M tokens, ratio 1.010).

## What I need from you

1. Load Part A in the harness and confirm the files parse (`prompts.jsonl` + `manifest.json`).
2. Review half of the manual items (plan rule): Hinglish Aya, XSTest HI/Hinglish, ClinicSafe-Hi. Mark `reviewed_by` or
   send me ids with comments.
3. Send me the clinic system prompt to review, and your Kaggle harness commit for the parity check (H.4).
4. Share the original-model FP16 answers for `hand_labels/sample_to_label.jsonl` so the κ labelling can start.

## Hub status

**Not pushed to the Hub.** The sets travel with this Git repo (`Datasets/final/`); weights and results via Google Drive. `scripts/data/package.py push` exists for when pushing is allowed again.
