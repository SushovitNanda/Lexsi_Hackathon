# Manish: November 7 independent work

## Completed: M11 medical protection dry run

Built both placeholder-protected and matched-random medical Q3_K_M variants directly from the same medical F16 checkpoint used on November 6. Synthetic seed-13 scores cover all 36 layers and seven matrix families. These scores are independent of response labels and are not a validated circuit map.

| Check | Result |
|---|---:|
| Stock medical Q3 size | 1,799,740,928 bytes |
| Placeholder-protected size | 1,795,390,976 bytes |
| Matched-random size | 1,795,390,976 bytes |
| Difference from stock, each | -0.2417% |
| Upgrades / downgrades, each | 8 to Q8_0 / 39 to Q2_K |
| Unselected tensors, byte-identical to stock | 243 per build |
| Stock/protected/random tensor names and shapes | All 290 match |
| EOS and template metadata | Preserved |

Both variants are within the +/-1% target and exactly equal in file size. The random control matches the original tensor shape/type classes and precision transitions, rather than only matching total size. All unchanged matrices have their measured stock types forced during export to prevent quantizer heuristic drift. Embedding, output and norm tensors are preserved.

The exporter now accepts `--artifacts-dir`, so this medical dry run writes its inventory, scores and tensor rules separately from November 5's original-model test. Earlier experimental evidence was preserved. The three protection tests pass. A separate benign eight-token load/tokenization/completion diagnostic is recorded in `results/mac/nov7/export_load_check.json`; it is not the Sunday real-map 100-prompt run-check, a utility score or a safety evaluation.

## Reproduce and inspect

```bash
.venv/bin/python scripts/mac/protect_gguf.py \
  --make-placeholder \
  --source models/gguf/med-f16.gguf \
  --stock models/gguf/med-q3km.gguf \
  --out models/gguf/medical-placeholder-nov7 \
  --report results/mac/nov7/medical_placeholder_protection.json \
  --artifacts-dir configs/mac/nov7
```

Full tensor inventory and the two 253-matrix export rule files: `configs/mac/nov7/`. Sizes, hashes, settings and selected tensors: `results/mac/nov7/medical_placeholder_protection.json`. Local model files: `models/gguf/medical-placeholder-nov7/{protected,random}-q3km.gguf`; weights remain ignored and are not pushed.

## What this means

This completes M11's mechanical medical-model dry run. It demonstrates that the exporter can allocate mixed precision and produce a matched random control within the file-size budget on the medical checkpoint. It does not demonstrate protection of safety, successful repair, circuit necessity or a clinical benefit. No new reported test responses or judge labels were generated for this task.

## Pending: M12 and external dependencies

M12 waits for Sushovit's D4, due November 7 at 20:00 IST: a 36-by-7 score table mapped to GGUF names and layer suggestions, with the circuit validation/G3a result, or the SafeTune neuron ranking if G3a fails. Existing upstream circuit/v2 score files are discovery outputs; no qualifying D4 validation handoff was found on origin/code_v2 commit 0f87f3f. They were not used here. On receipt, validate provenance, full tensor coverage, finite scores and the approved source/ranking before generating real protected/random rules. Sunday M13 builds and its nonreported 100-prompt run-check follow those real rules.

Current reference merge parity, the >=48/50 shared judged fidelity gate and centralized labels remain unresolved. They do not prevent this mechanical dry run, but safety claims and final analysis remain provisional. The five November 6 response ZIPs are already on remote branch manishshaw. This report, rules, inventories and checks are also shared on that branch; no message has been sent to Sushovit.

## Completed ahead of November 8: M14 and M13 preparation

M14's RQ5/RQ6 article methods and ROI method are drafted in `reports/methods/rq5_rq6_roi.tex`, with a visually checked three-page PDF and standalone LaTeX under `output/pdf/manish_rq5_rq6_roi_methods.*`. The ROI calculator (`scripts/mac/roi.py`) separates API cost, hardware amortization, energy/operating cost, monthly capacity and break-even; six tests pass. All example prices, throughput and clinic demand are explicitly invented arithmetic inputs, not measured ROI or vendor quotes. Actual ROI results remain pending valid workload/termination/power measurements and dated pricing.

M13's calibration-only 100-prompt operational set is fixed, reproducible and disjoint from test IDs and normalized text. Its manifest is `configs/mac/planning/m13_runcheck100.json`; the local prompt file is ignored. `reports/methods/M13_READINESS.md` supplies the D4 acceptance checklist and conditional build/check commands. This prepares M13 but does not complete it: no real-map models or 100-prompt real-map checks were run without D4.
