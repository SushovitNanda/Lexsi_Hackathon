# Manish: November 4-5 execution record

Scope: plan v3, Manish M1-M8. Yesterday's setup and evidence are recorded in `MAC_TODAY.md`. No future-day medical/repaired test cells are included here.

| Step | Current evidence |
|---|---|
| M1-M4 | Mac environment/Metal build, Global and Base downloads, original GGUFs, template/EOS checks, and 200-prompt dev benchmark complete. |
| M5 | Medical/control CPU FP32 merges saved as all-finite FP16; current D2 reference parity remains unverified. |
| M6 | Both original-model cells complete: 1,612 responses each, zero errors. Official: 3,560.35 seconds / 319,965 generated tokens; no-preamble: 2,776.26 seconds / 310,453 tokens. Identical IDs, input digests, model hash and controlled settings verified. |
| M7 | Medical F16 GGUF converted, hashed and template-audited; all 50 fixed calibration responses generated with zero request failures (193.53 seconds). The T4/central-judge comparison remains external. All 50 hit the token cap; the separate termination diagnostic below is included for review. |
| M8 | Real placeholder protected/random GGUF builds passed size, type, metadata and byte-integrity checks. |

## Frozen protocol

`MAC_PREREG_2026-11-05.md` and `configs/mac/nov5_inputs.json` were committed as **2582b4d** before test generation. The v3 plan allows using its protocol while D3 is pending. The older `PREREG.md` remains Sushovit's draft. No co-signature, final shared Holm family, judge calibration or cross-machine fidelity pass is implied by this Mac registration.

The new test harness requires a committed, unchanged registration; captures implementation/model/template/input hashes; validates all prompt token IDs and context lengths before generation; and checkpoints each response by prompt ID. Raw working response files remain ignored under `models/local-results/day2/`; the packaged ZIP was later force-added and pushed at the user's explicit request (see handoff status below). Output line order follows completion order: join on `(cell_id, prompt_id)`.

## M8 mechanics and reproduction

Stock Q3_K_M has 290 tensors: 37 F32, one Q6_K, 144 Q3_K, 104 Q4_K and four Q5_K. `results/mac/stock_q3km_tensors.json` records every tensor's shape/type/bytes. Only 252 transformer matrices are eligible for changed precision; embedding and norm tensors are preserved.

The seed-13 placeholder map is synthetic and has not passed RQ3 validation. Eight high-score tensors were raised to Q8_0 and 39 low-score tensors lowered to Q2_K. The random control matches each original-type/shape/new-type transition. Both actual files are **1,795,391,040 bytes**, exactly equal and **0.2417% below** stock's 1,799,740,992 bytes. In each build, all 243 unselected tensors are byte-identical to stock; all tensor types/shapes, EOS and template metadata are verified. This establishes mechanical size-budget compliance, not a safety benefit.

The exporter forces the measured baseline type of every weight matrix before applying selected changes. Partial overrides alone can shift llama.cpp's default K-mixture counters; the initial test detected that unintended change, and the failed engineering artifact is preserved privately with its diagnostic log. The final rules avoid it.

```bash
.venv/bin/python scripts/mac/protect_gguf.py --make-placeholder
.venv/bin/python -m pytest tests/test_mac_harness.py tests/test_mac_protection.py -q
```

Rules: `configs/mac/placeholder_protected.tensor-types.txt` and `placeholder_random.tensor-types.txt`. Results/hashes: `results/mac/placeholder_protection_report.json`. Builds: ignored `models/gguf/placeholder/`. To export a real table later, pass `--scores` with the agreed 36-layer x 7-matrix `layer_x_matrix` JSON and the corresponding source/stock files. Do not substitute the old, unvalidated circuit pass for D4/G3a or claim these placeholder builds as RQ6 evidence.

## M7 fidelity and D6

The proposed shared fidelity set is frozen before responses: 18 English, 16 Hindi, 16 Hinglish; 25 harmful and 25 benign calibration items. No agreed replacement set or current T4 responses has arrived. The medical GGUF source is the local all-finite FP16 merge. Its provenance is `results/mac/medical_gguf_build.json`; the template audit is `results/mac/medical_templates/audit.json`.

`scripts/mac/fidelity_t4_reference.py` is a standalone helper for Sushovit's Colab T4. It checks the exact input digest/IDs, current local merge hashes and FP16 dtype, then generates greedy 256-token responses with EOS 3. It requires a T4; it is not run on this Mac. Both sets of responses must be judged on the same pinned judge host. The v3 fidelity criterion is >=48/50 matching labels; it remains pending until the paired labels and merge-reference check exist.

`scripts/mac/package_day2.py` validated IDs/counts/fingerprints and created **`models/local-results/day2/D6_Manish_2026-11-05.zip`** with all **3,274 responses**, fixed inputs, manifests, protocol, reference helper and centralized judging instructions. The user subsequently requested a remote push, and the ZIP was uploaded to the public `origin/manishshaw` branch in commit **cef0e55**. Direct receipt/judging by Sushovit is not confirmed. The ZIP's internal transfer status records its earlier creation-time state; the current upload status is recorded in `results/mac/day2_handoff.json`. All nine targeted software tests pass; native placeholder builds were checked independently.

### Medical termination observation

All 50 medical generations used the complete 256-token budget. A separate benign 64-token probe produced a normal short answer followed by 51 hidden end-of-turn tokens (ID 6), one end-response token and no EOS token (ID 3). The training chat template ends assistant turns with 6, while the inherited generation config uses 3. Original and medical GGUF architecture metadata, including RoPE, match. This is a termination mismatch to review with Sushovit's T4 output; it is not evidence that all visible answers were truncated or that the medical GGUF is numerically faithful. The frozen 50 responses were retained, and no stop rule or response selection was changed after observing them. The T4 helper now records token counts, final token and end-turn counts. Evidence: `results/mac/medical_termination_diagnostic.json`, also inside the private hand-off. Review before reported medical cells and disclose any protocol amendment.

## Remaining external hand-offs

- Current medical/control `merge_info.json`; reference tensors if current hashes differ and maximum absolute difference must be measured. Historical-log hash differences have been disclosed; hashes alone do not give numerical differences.
- Confirmation of the exact 50 fidelity prompts, Sushovit's medical FP16 T4 responses and centralized judge labels after G1.
- Confirmation of Sushovit's receipt and judging of the uploaded D6 archive. The original plan's no-push rule was superseded by the user's explicit instructions to push the code, plan PDF and ZIP.
