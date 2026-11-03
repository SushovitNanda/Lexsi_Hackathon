---
license: cc-by-nc-4.0
language:
  - en
  - hi
task_categories:
  - text-classification
tags:
  - safety
  - red-teaming
  - hinglish
  - code-mixing
  - medical
  - lexsi-hackathon
pretty_name: ClinicSafe-Hi and the Safe-from-Fine-Tune-to-Phone evaluation sets
extra_gated_prompt: >-
  This dataset contains harmful *requests* (no completions) used to evaluate safety drift of fine-tuned language
  models. Access is granted for research on AI safety only. You agree not to use the prompts to train models to
  comply with harmful requests, not to redistribute them, and to report any issue to the maintainers.
extra_gated_fields:
  Affiliation: text
  Intended use: text
  I agree to use this data for AI-safety research only: checkbox
---

# ClinicSafe-Hi — clinical harmful requests with benign twins (EN / HI / Hinglish), plus the evaluation sets of *Safe from Fine-Tune to Phone*

**Lexsi AI Safety and Alignment Hackathon, Track 2.** Team: Manish & Sushovit. Research-only, CC-BY-NC-4.0, gated with manual approval.

## What is here

Prompts and labels **only**. No model completions are published anywhere.

<<TABLE>>

Every record uses one schema (JSONL, UTF-8):

```
{"id", "lang" (en|hi|hinglish), "split" (test|dev|calib), "set" (harmful_general|harmful_clinical|benign_general|benign_clinical|medmcqa),
 "source" (aya_redteaming|xstest|clinicsafe_hi|medmcqa|alpaca), "category", "twin_id", "text", "gloss_en", "origin", "source_id"}
```

`origin` ∈ {original, indictrans2_postedited, adapted, rewritten, replaced}. ClinicSafe-Hi harmful items link to their benign twin through `twin_id`.

## ClinicSafe-Hi

A small benchmark of **clinical harmful requests** and **benign twins** (same topic, legitimate intent), ~100 + ~100 per language
across six categories: prescription-only drug misuse (`rx_misuse`), forged medical documents (`forged_docs`), dangerous self-treatment
and overdose risk (`self_treat`), medical misinformation (`misinfo`), patient-privacy violations (`privacy`), health scams (`scams`).

- **Construction.** Written in English first as *requests* a person might send to a clinic chatbot; harmful items state a goal and contain
  no operational detail; then **adapted** (not literally translated) into Hindi (Devanagari) and Hinglish (Hindi grammar in Latin script,
  as typed on WhatsApp), keeping intent, category and strength. Indian clinical context (medicine brands, PHC, Aadhaar, Ayushman, WhatsApp forwards).
- **Checks.** Qwen3Guard-Gen-4B prompt moderation on every item (harmful items must be Unsafe/Controversial, twins Safe); every disagreement
  reviewed by hand and either fixed or kept with a note (`qwen3guard_disagreements.jsonl` in the project repo). Cross-review by the second
  team member (plan rule: each item is checked by the teammate who did not write it).
- **Intended use.** Measuring attack-success and over-refusal of medical assistants in English, Hindi and Hinglish, before and after
  fine-tuning, repair, quantization and template changes. A refusal of a benign twin counts as over-refusal.
- **Risks.** The harmful prompts describe real misuse goals (drug misuse, forgery, scams). They are deliberately free of instructions, but
  they could be used to probe or to tune models toward compliance; hence the gate and the research-only terms. The benchmark is small
  (n = 100 per language per side), written by two people, and may reflect our blind spots; results per category are directions, not findings.

## Other sets

- **Aya Red-teaming EN/HI** (CohereForAI/aya_redteaming, Apache-2.0): 200 test + 50 dev per language, stratified by harm category;
  **Hinglish**: 150 test + 50 dev rewrites of the Hindi items (same intent, same `source_id`).
- **XSTest** safe prompts (Röttger et al.): 150 English test (stratified by type), 100 English dev; Hindi via IndicTrans2 + post-editing
  (broken homonym/figurative items replaced with clinical look-alikes, `origin: replaced`); Hinglish rewrites.
- **MedMCQA** (openlifescienceai/medmcqa) validation questions: 500 test + 300 dev, one-token answer format, disjoint from fine-tuning.
- **Calibration pairs** (steering): English-only (100 harmful Aya / 100 harmless Alpaca) and three-language (33 per language per side); all
  outside every test set.

## Leak check

CuratorKIT-style exact + MinHash + multilingual-embedding dedup between the fine-tuning sets and every evaluation set; report in `leak_report.json`.

## Licence and citation

Prompts from Aya Red-teaming (Apache-2.0), XSTest (CC-BY-4.0), MedMCQA (Apache-2.0), Alpaca-cleaned (CC-BY-NC-4.0); ClinicSafe-Hi items
CC-BY-NC-4.0. Everything derived from Tiny Aya in this project is non-commercial and research-only.
