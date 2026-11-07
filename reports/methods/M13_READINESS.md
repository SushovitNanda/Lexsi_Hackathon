# M13 readiness: real-map medical protected/random builds

Prepared November 7 for the November 8 operational check. **Prepared, not executed.** November 7 placeholder builds completed M11; they do not satisfy M13's real-map requirement.

## D4 acceptance checklist

Before execution, record Sushovit's handoff source/revision and checksum, model identity, G3a pass evidence (necessity against all five random sets in at least one language), and approved language aggregation/normalization. If G3a fails, require the identified SafeTune fallback and the approved mapping from neuron ranks to tensor scores. Do not infer that discovery tables are validated, invent scores for absent matrices, or average language tables without approval of that recipe.

The exporter expects `layer_x_matrix`, a list of exactly 36 uniquely numbered layers 0-35, each containing finite numeric scores for `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`. Current solver ranks absolute score magnitudes; confirm that this is appropriate to the supplied scores. GGUF names are `blk.<layer>.<matrix>.weight`. Require the layer suggestions alongside the score table. Preserve the raw handoff and separately document any approved schema conversion. Code accepting a table's shape does not prove circuit validation.

Freeze count/type choices and seed before test generation. The mechanical operating point is 8 Q8_0 upgrades and sufficient Q2_K donors under the exact stock layout, with seed-13 matched random selection. If D4 requires a different operating point, document it prospectively from development evidence, never test responses.

## Fixed operational inputs

Run `.venv/bin/python scripts/mac/prepare_m13.py` to reproduce `configs/mac/planning/m13_runcheck100.json` and the ignored local prompt file `models/local-results/day5/runcheck100_prompts.jsonl`. The manifest fixes 100 calibration prompts: EN34/HI33/Hinglish33, harmful50/benign50, and no test-ID or normalized-text overlap. Regeneration fails if the frozen manifest changes. Calibration overlap with prior fidelity/dev inputs is permitted for this nonreported operational check; it is not independent safety evidence.

## Commands after D4 is accepted

These commands are preparation only. `configs/mac/d4/approved_scores.json` does not exist yet and must come from the accepted D4 recipe.

```bash
.venv/bin/python scripts/mac/protect_gguf.py \
  --scores configs/mac/d4/approved_scores.json \
  --source models/gguf/med-f16.gguf \
  --stock models/gguf/med-q3km.gguf \
  --out models/gguf/medical-d4 \
  --report results/mac/d4/medical_protection.json \
  --artifacts-dir configs/mac/d4/export
```

Require source/stock hashes to match November 6, exact protected/random size equality, both within +/-1% of stock, all expected tensor names/shapes/types, preserved template/EOS and byte identity for unselected tensors. If export fails, retain its report/logs and diagnose the failure; do not use placeholder outputs as substitutes.

Start one server at a time, replacing `<tag>` with `protected` or `random`:

```bash
.venv/bin/python scripts/mac/serve.py \
  --gguf models/gguf/medical-d4/<tag>-q3km.gguf \
  --log models/local-results/day5/<tag>.server.log
```

In a second terminal, run the fixed diagnostic (calibration split, not a reported test cell):

```bash
.venv/bin/python scripts/mac/run_cell.py \
  --cell med_d4_<tag>_runcheck100 \
  --gguf models/gguf/medical-d4/<tag>-q3km.gguf \
  --tokenizer models/merged/med \
  --server-receipt models/local-results/day5/<tag>.server.receipt.json \
  --protocol MAC_PREREG_2026-11-06.md \
  --split calib \
  --inputs models/local-results/day5/runcheck100_prompts.jsonl \
  --out models/local-results/day5/<tag>_runcheck100.jsonl
```

Stop the first server cleanly before loading the other; verify its receipt's native PID/command before signaling it. Completed checks require 100 exact IDs, no duplicates, matching input digests, one fingerprint per cell and zero final execution errors. On interrupted resume, use `resume_text_cell.py` instead of `run_cell.py` only for its explicitly audited randomized-media-marker compatibility path; all other changes must still fail the frozen resume check.

Retain run specs, completion-token diagnostics and execution summaries. Do not compute/report ASR from this operational set, choose tensor rules from its answer quality, or interpret lack of execution errors as safety. Cross-machine parity, the judged fidelity50 gate, and D7 labels remain separate unresolved dependencies. Reported repaired protection additionally waits for D5.
