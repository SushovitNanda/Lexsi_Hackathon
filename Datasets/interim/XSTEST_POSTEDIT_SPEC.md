# XSTest Hindi post-edit, replacements and Hinglish (Task_1 I.6.2–I.6.4)

Input sheet: `Datasets/interim/xstest_en150_sheet.jsonl`, one line per item with
`id, english, type, focus, note, source_id, hindi_draft (IndicTrans2), hindi, hinglish, origin, gloss_en, replaced_english`.

XSTest *safe* prompts look alarming but are harmless (homonyms, figurative language, safe targets, safe contexts,
definitions, discrimination questions about non-real groups, historical events, privacy of public figures / fictional
characters). The set measures **over-refusal**: a model that refuses them is being over-cautious.

## Rules

1. **Post-edit the draft** (`origin: "indictrans2_postedited"`): make the IndicTrans2 Hindi read like natural spoken
   Hindi, keep the meaning, keep the "sounds alarming but is safe" quality, fix wrong word senses and mangled names.
   Common English loanwords may stay in Devanagari (कंप्यूटर, प्रोसेस, पार्टी, मूवी).
2. **Replace broken items** (`origin: "replaced"`): if the translation loses what made the prompt safe-but-alarming
   (e.g. "kill a Python process" is safe only because of the English word; many figurative idioms do not exist in Hindi),
   write a **clinical look-alike** instead: a medically safe question that sounds alarming, e.g. about lethal doses in the
   abstract, drug interactions, "killing" bacteria/parasites/cancer cells, "poisonous" plants for a child's safety,
   what a toxic blood level is, how long a person can survive without water, overdose warning signs, etc. Record the
   original English in `replaced_english`, write the *new* English meaning in `gloss_en`, and keep `type` as it was.
   Replace only when needed; aim for ≤ 25 % replacements overall.
3. Items that are not broken but only need light adaptation (names, places) get `origin: "adapted"`.
4. **Hinglish** (`hinglish` field): rewrite the final Hindi item in Hinglish per `HINGLISH_STYLE.md` (Latin script,
   Hindi grammar, English words where natural). For replaced items, write the Hinglish of the replacement.
5. `gloss_en`: the English meaning of the final Hindi item (equal to `english` unless replaced).
6. Keep every item a single, short question or request (as XSTest items are). No answers.
7. Output: the same lines in the same order, all fields filled: `hindi`, `hinglish`, `origin`, `gloss_en`
   (`replaced_english` only for replaced items, else null).
