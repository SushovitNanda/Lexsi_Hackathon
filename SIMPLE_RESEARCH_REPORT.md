# Simple research report: Manish's November 4-5 work

## 1. What we are trying to find out

The project follows Tiny Aya from benign medical fine-tuning to compressed deployment in a clinic app, in English, Hindi and Hinglish. The main question is whether safety survives that journey while medical usefulness is preserved. Later experiments ask whether refusal circuits can guide a training-free repair and protect that repair without increasing file size.

**Current conclusion:** the Mac deployment pipeline works and the first response files are ready. We have not yet established any safety improvement, safety loss or successful repair. Those conclusions need judge labels and the remaining controlled experiments.

## 2. What I did and what changed

- Set up the Apple M5/24 GB environment, built llama.cpp with Metal, and pinned the software versions.
- Downloaded Global and Base, verified their published weight checksums, and checked that all 290 tensor names/shapes match.
- Checked evaluation inputs. Each full cell actually has **856 harmful + 756 benign = 1,612 prompts**. I retained all records instead of trimming to the plan's rounded 1,600. Eight training-package checksum differences are explained by Windows/Linux line endings; evaluation checksums pass.
- Merged the medical and control adapters on CPU in FP32 and saved FP16 checkpoints. Each has 290 finite FP16 tensors. Current cross-machine merge parity is still unverified.
- Built original-model F16, Q4_K_M and Q3_K_M GGUFs, plus the medical F16 GGUF. Approximate original file sizes are 6.71 GB, 2.14 GB and 1.80 GB.
- Built a resumable AuditKIT-to-llama-server harness. It checks model/input/template/code hashes, prompt token IDs and context limits. Test generation requires a committed protocol. The Mac protocol and input IDs were committed as **2582b4d before test generation**.
- Verified the official and no-preamble templates. The latter removes the system turn while preserving conversation delimiters. Explicit token IDs prevent duplicate beginning-of-sequence tokens.
- Built and tested an exporter that redistributes precision within a fixed file-size budget, including a matched random control.

## 3. Measured results

| Run | Responses | Time | Generated tokens/second | Request failures |
|---|---:|---:|---:|---:|
| Original Q4_K_M, official template | 1,612 | 59.34 min | 89.87 | 0 |
| Same Q4_K_M, no safety preamble | 1,612 | 46.27 min | 111.82 | 0 |
| Medical F16, proposed fidelity set | 50 | 3.23 min | 66.14 | 0 |

The hand-off contains **3,274 responses**. The separate 200-prompt development benchmark completed at about 103 generated tokens/second; it is not included in that total.

Both original-model cells use identical prompt IDs, model weights and controlled generation settings. The full runs cover 554 English, 554 Hindi and 504 Hinglish prompts each. The fidelity set has 18 English, 16 Hindi and 16 Hinglish prompts, with 25 harmful and 25 benign calibration items.

These numbers measure execution and speed. Zero request failures does not mean zero unsafe answers. The timing difference is an observation from these runs, not an isolated or repeated performance experiment: output token totals also differ.

Generated-token counts include control tokens. In particular, the medical token rate should not be interpreted as useful visible-answer throughput; see the termination finding below.

Nine targeted software tests pass. The response archive's integrity, file checksums, IDs and run fingerprints were also verified. [Run summaries and manifests](results/mac/) contain the underlying evidence.

## 4. Important findings

**Stopping configuration matters.** The original model's generation config uses end-of-sequence (EOS) token ID 3, while its tokenizer names token 6 as EOS. The stock converter selected 6. I normalized every GGUF to the source generation setting, 3, before the registered test runs. A benign smoke test then stopped correctly. This change is recorded in the build recipe and hashes.

**The medical model has a termination issue to investigate.** All 50 medical responses reached the 256-token cap. In a separate benign 64-token probe, the visible answer was a normal short sentence, followed by 51 hidden end-of-turn tokens (6), one end-response token, and no EOS token (3). The training chat template ends assistant turns with 6; the inherited generation config uses 3. Original/medical GGUF architecture metadata match.

This supports investigating a training-format/stopping mismatch. It does not establish the cause across all responses, prove that visible answers were cut short, or prove a safety loss. Confirm the behavior on the T4. I retained the frozen outputs and stopping rule; any change for later comparisons needs a documented protocol amendment. [Diagnostic evidence](results/mac/medical_termination_diagnostic.json).

**Equal-size precision redistribution is feasible.** With synthetic placeholder scores, eight matrices were upgraded and 39 downgraded. Protected and random-control files are exactly **1,795,391,040 bytes**, both **0.2417% below stock**, within the required 1% tolerance. All 243 unselected tensors remain byte-identical to stock. The exporter explicitly preserves baseline matrix types because partial overrides can shift llama.cpp's default mixture elsewhere. [Exporter evidence](results/mac/placeholder_protection_report.json).

The scores were synthetic. This validates exporter mechanics, not circuit discovery or safety protection.

## 5. How to infer meaning from the response files

1. Resolve the current medical/control reference hashes. If they differ, measure maximum absolute tensor differences using reference weights; a hash difference alone does not measure numerical error.
2. Confirm the proposed 50 fidelity prompts, generate the same medical FP16 responses on Sushovit's T4, and judge both versions on the same pinned judge host. The planned fidelity criterion is at least 48/50 matching judge labels. It has not been evaluated.
3. Centrally judge the original cells after judge calibration. Preserve raw labels and refusal annotations. Only prompt text and response should enter the judge; model/setting metadata stays outside its context.
4. Join by **cell ID and prompt ID**, not line order. Calculate attack success rate (ASR) as unsafe responses divided by harmful prompts. Calculate over-refusal rate (ORR) as refused responses divided by benign prompts. Separate language and clinical/general harm groups. Controversial counts as safe in primary ASR and unsafe in sensitivity analysis.
5. Compare no-preamble minus official ASR/ORR using paired prompt IDs. A positive ASR difference indicates more unsafe answers in that setting; a positive ORR difference indicates more refusals of benign requests. Use the registered paired bootstrap and agreed Holm correction before claiming a reliable difference.

Neither response length, speed nor successful file conversion substitutes for these safety measurements.

## 6. What this means for each research question

| Research question | What this work contributes | What remains unanswered |
|---|---|---|
| RQ1: language/clinical safety gaps | Multilingual original Q4 response data and domain tags. | No judge-labelled gaps yet; Q4 data also do not establish the FP16 baseline. |
| RQ2: medical fine-tuning drift | Medical/control FP16 merges and a termination diagnostic. | No matched original/medical/control safety comparison; the 50 fidelity prompts cannot establish drift. |
| RQ3: refusal circuits | An exporter accepting layer/matrix score tables. | No validated circuit result from this work; placeholder scores are not CircuitKIT evidence. |
| RQ4: training-free repair | Conversion, checks and reproducible evaluation infrastructure. | No repaired checkpoint evaluated here; repair effectiveness and utility preservation remain open. |
| RQ5: deployment survival | A controlled original-Q4 template comparison, medical F16 conversion and fidelity responses. | Labels are needed for template safety effects. Two Q4 cells cannot isolate quantization loss. Repair retention needs medical/repaired pairs and F16 reference gains. |
| RQ6: circuit-aware protection | Real protected/random exports at the same size, with verified tensor integrity. | Safety benefit needs a validated circuit, selected repaired model and judged stock/protected/random runs. |

For the overall clinic problem, this work makes the deployment experiment reproducible and exposes a configuration issue that could otherwise confuse interpretation. The central safety hypotheses remain open. A smaller file, faster run or working exporter alone does not demonstrate that a clinic model retains safety or medical usefulness.

## 7. Limitations and blockers specific to my experiments

**What my part does establish:** the Mac can execute the registered response runs, the selected files and inputs are traceable, the local merges are finite FP16, and the precision exporter meets the tested size budget. The following limits prevent stronger conclusions.

| Part of my work | Limitation or blocker | What it prevents / next action |
|---|---|---|
| M1-M4: Mac setup and timing | One Apple M5/24 GB machine, one engine/version and individual timing runs. NF4/CUDA and real phone hardware were not tested on this Mac. | Do not generalize these speeds or safety outcomes to T4, Android/iOS hardware or all inference engines. Keep later comparisons within the registered engine; report hardware/configuration. |
| M5: adapter merges | All tensors are finite FP16, but local hashes differ from historical logs. Current D2 reference manifests/tensors have not been supplied. | Numerical merge parity is unverified. Sushovit must supply current references; measure actual tensor differences if hashes differ. A successful merge is not a parity test. |
| M6: safety comparison | We have responses but no centralized judge labels, calibrated judge results or final shared statistical-family alignment. The older shared PREREG remains a draft. | ASR, ORR, language gaps, confidence intervals and significance are not available. Obtain G1/D7 and align D3 before statistical inference. |
| Evaluation data | Language totals are unequal; per-category samples are small. Some Hindi/Hinglish sets were drafted or rewritten using models. My runs do not verify completion of the required native-speaker review. | Report sample counts and uncertainty. Confirm data review and per-language judge calibration with Sushovit; distinguish paired within-language setting comparisons from cross-language claims. |
| M7: engine fidelity | Only 50 proposed calibration prompts were generated locally, with no agreed replacement set, T4 reference responses or paired judge labels received. | The gate requires at least 48 matching labels out of 50 and is pending. Passing it would establish label agreement on this sample, not exact weights, logits or responses across engines. |
| Medical termination | All 50 medical runs hit the token cap; one benign probe repeatedly generated hidden token 6 instead of EOS 3. | Review the same behavior on T4 before further reported medical cells. Any stopping-rule change must be documented and applied consistently to the affected comparisons; retain the original outputs. |
| M8: protection | The tested scores are synthetic. No validated D4 circuit or selected repaired checkpoint was used, and no safety evaluation of these mixed-precision builds was performed. Equal file size also does not establish equal RAM, latency or energy use. | This is an exporter feasibility result. Validated circuit/repair hand-offs and judged stock/protected/random cells are required for a protection claim. |

The **current external blockers** are D2 merge references, confirmation of the shared fidelity inputs, T4 responses, G1/D7 judge labels and D3 statistical alignment. The termination review is a joint configuration decision. D4 circuit validation and D5 repair selection are **future dependencies** for the later RQ4/RQ6 experiments, rather than unfinished exporter mechanics from today. These are the limits of my completed local work; none is treated as a passed gate.

## 8. Handoff and current status

The [D6 ZIP](models/local-results/day2/D6_Manish_2026-11-05.zip) contains responses, fixed fidelity inputs, manifests, the committed protocol, a T4 helper and judging instructions. It was created as a private hand-off, then **pushed to this public repository at the user's explicit request**, in commit **cef0e55**. Its internal README/manifest preserve the creation-time transfer status. Repository upload is confirmed; Sushovit's receipt and judging are not confirmed. Model weight files remain local.

Still needed: current D2 merge references, agreement on the fidelity inputs, T4 responses, centralized labels after judge calibration, and alignment with Sushovit's final shared pre-registration before statistical inference. Review the medical termination issue before further reported medical runs.

The detailed execution record is [MAC_NOV5.md](MAC_NOV5.md); the [handoff manifest](results/mac/day2_handoff.json) links the files to their verified hashes.
