# Manish — Wednesday 4 November (M1–M5)

Plan: `Lexsi_Labs 2.pdf`, v3, pages 8–11. This records the local work. Code, the plan PDF and the D6 response ZIP were subsequently pushed at the user's explicit request; see `SIMPLE_RESEARCH_REPORT.md` for the current upload status. Model weights and raw working response files remain local.

| Step | Verified status |
|---|---|
| M1 | Python 3.12 environment installed and locked; llama.cpp built with Metal; Apple M5 detected. Pins appended to ENV.md. |
| M2 | Global and Base downloaded at pinned revisions; all source shard hashes match published checksums. Base access resolved and download verified on 5 November; all 290 tensor names/shapes match Global. HF/GGUF template audit passed; live official and no-preamble server renderings checked with BOS normalization. |
| M3 | Original F16, Q4_K_M and Q3_K_M GGUFs built and hashed. `templates/none.jinja` removes exactly the system turn. Generation EOS normalized to source ID 3 in every GGUF (details below). |
| M4 | AuditKIT API harness implemented. Real Metal dev benchmark: 200/200 responses, no errors, 342.14 seconds, 35,226 generated tokens, 102.96 tokens/s aggregate. Six regression tests pass. |
| M5 | Evaluation package verified; medical/control/2-epoch adapters inventoried. Medical and control adapters merged on CPU in FP32, saved FP16; all 290 tensors in each output are finite F16. Current D2 reference manifests/weights are missing, so cross-machine merge parity remains unverified. |

## Evidence and discrepancies

- `results/mac/environment.json`: actual machine, package versions, llama.cpp commit and detected devices.
- `results/mac/data_verification.json`: evaluation checksums pass. Actual full-cell counts are **856 harmful + 756 benign = 1,612**, plus 500 MedMCQA for the separate accuracy task. Preserve all 102 clinical pairs per language; do not silently trim to the PDF’s rounded counts.
- Eight SFT checksum mismatches are explained exactly by CRLF versus LF conversion. Source files/checksums have been preserved; ask Sushovit to reconcile raw-byte checksums before publishing. Both `train.jsonl` checksums pass.
- `results/mac/templates/audit.json`: English/Hindi/Hinglish HF/GGUF renderings, token counts and template hashes. Live server checks are in `server_smoke.json` and `server_none_template_audit.json`; `/apply-template` omits the BOS that the server's string tokenizer adds, so compare with this normalization. Explicit raw token IDs retain exactly one BOS.
- `results/mac/adapter_inventory.json`: actual adapter file hashes, finite-weight checks, tensor shapes/dtypes and training summaries. These are local inventory hashes, not Sushovit’s expected merge hashes.
- `results/mac/original_gguf_build.json`: source revision, published/verified shard hashes, llama.cpp commit, final GGUF sizes and hashes.
- `results/mac/download_status.json`: successful Global/Base downloads and checksum verification; Base revision and all 290 tensor names/shapes checked against Global on 5 November. Local snapshot paths are in ignored `models/gguf/source_snapshots.json`.
- `results/mac/benchmark_dev200.json`: measured dev timing only, not a reported test cell. Responses remain private under ignored `models/local-results/`.
- `results/mac/merge-med-check.json` and `merge-ctrl-check.json`: actual FP16/finite checks and hashes. `historical_merge_comparison.json` compares the older committed logs; both local builds differ from those historical hashes. Numerical differences cannot be determined from hashes; current reference weights are required if current reference hashes also differ.
- The checked-in `PREREG.md` is still a v2-era draft. No reported test cell was run; align it with v3 and Sushovit’s final cell IDs before test generation.

### Source generation EOS versus tokenizer EOS

The source `generation_config.json` uses EOS ID **3**, while its tokenizer names EOS ID **6**. The stock converter selects 6, causing the raw GGUF engine to continue past the model's generated `<EOS_TOKEN>` (3). The preparation recipe now normalizes only this metadata value to **3** in every GGUF and hashes the final files. Tensor bytes and file sizes are unchanged by normalization, covered by a regression test. The live smoke response now ends at EOS after 11 tokens. Record this convention for all subsequent stock, repaired and protected builds; it is part of the deployment method, not a quantization intervention. Evidence: `results/mac/gguf_eos_alignment.json`.

## Finish the model-dependent steps

Both model downloads are complete (Base verified 5 November). The remaining dependency is Sushovit’s current D2 reference manifest (reference weights only if hashes differ). The commands below reproduce the completed work. Credentials are read locally and never written to tracked files.

```bash
export HF_HOME="$PWD/models/hf_cache"
export PYTHONUTF8=1
# Already logged in; only repeat if credentials expire:
.venv/bin/hf auth login

# Verify cached Global/Base and final GGUF outputs. On a fresh checkout,
# this also converts F16, quantizes, aligns EOS and audits the templates.
.venv/bin/python scripts/mac/prepare_models.py

# Get the exact source snapshot path written by prepare_models.py.
TOKENIZER_PATH=$(.venv/bin/python -c 'import json; print(json.load(open("models/gguf/source_snapshots.json"))["global"]["path"])')

# Run in terminal 1; wait for Ready. Stop with Ctrl-C after the benchmark.
.venv/bin/python scripts/mac/serve.py --gguf models/gguf/orig-q4km.gguf
```

In terminal 2:

```bash
TOKENIZER_PATH=$(.venv/bin/python -c 'import json; print(json.load(open("models/gguf/source_snapshots.json"))["global"]["path"])')
.venv/bin/python scripts/mac/run_cell.py \
  --cell smoke-orig-q4km-dev200 --gguf models/gguf/orig-q4km.gguf \
  --tokenizer "$TOKENIZER_PATH" --server-receipt models/local-results/server.receipt.json \
  --split dev --limit 200 --workers 8 \
  --out models/local-results/smoke-orig-q4km-dev200.jsonl
```

The receipt verifies the live server command, model hash, cache settings and model path. Each prompt’s HF token IDs must match llama.cpp’s tokenizer; explicit token IDs prevent double BOS insertion. Server prefix reuse is disabled for reproducibility. The server has 8 slots and 16,384 total context (2,048 per slot). Overlong prompts fail visibly. The summary records measured elapsed time, responses/s and generated tokens/s. Repeating the exact command resumes missing IDs; changed settings or inputs refuse to reuse the file. Interrupted final JSONL fragments are saved as `.partial`. Responses stay under ignored `models/` for private D6 hand-off.

On 5 November the harness added implementation hashes and a committed-protocol requirement for test cells. The original dev benchmark remains frozen evidence from the earlier implementation. Choose a new output filename when rerunning that benchmark with the newer code; do not overwrite its original measurements. Test-cell resumes require identical code as well as identical inputs/settings.

For the no-preamble cell use `--template none`. For the later phone-cache cell restart `serve.py --phone-cache`, and pass `--cache-k q4_0 --cache-v q4_0` as well as `--template none`. Template selection is local, via raw `/v1/completions`; it cannot be silently replaced by a server chat template. Do not treat the dev timing run as a reported test cell.

Reproduce the completed merges sequentially using the exact local Global snapshot:

```bash
export HF_HOME="$PWD/models/hf_cache"
GLOBAL_PATH=$(.venv/bin/python -c 'import json; print(json.load(open("models/gguf/source_snapshots.json"))["global"]["path"])')
GLOBAL_REV=$(.venv/bin/python -c 'import json; print(json.load(open("models/gguf/source_snapshots.json"))["global"]["revision"])')
.venv/bin/python finetune/merge.py runs/med/adapter models/merged/med --base-model "$GLOBAL_PATH" --base-revision "$GLOBAL_REV"
.venv/bin/python finetune/merge.py runs/ctrl/adapter models/merged/ctrl --base-model "$GLOBAL_PATH" --base-revision "$GLOBAL_REV"

# Replace the expected path with Sushovit’s D2 reference manifest.
.venv/bin/python scripts/mac/check_merge.py --model models/merged/med \
  --expected /path/to/Sushovit/med/merge_info.json --out results/mac/merge-med-check.json
```

`check_merge.py` checks saved dtypes, finite values and hashes. Omit `--expected` for local validation only (it records parity as unverified). If reference hashes differ, optional `--reference /path/to/reference/weights` measures the maximum absolute tensor difference; hashes alone cannot establish that difference. Do not compare a generated local manifest against itself as evidence of parity.

## Reproduce setup and checks

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-mac.lock
git clone https://github.com/ggml-org/llama.cpp.git KITS/llama.cpp
git -C KITS/llama.cpp checkout 5851a8e505f4d5fbb183f2847698eb6155d176b7
PATH="$PWD/.venv/bin:$PATH" cmake -S KITS/llama.cpp -B KITS/llama.cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
PATH="$PWD/.venv/bin:$PATH" cmake --build KITS/llama.cpp/build \
  --target llama-server llama-quantize llama-cli -j 8
.venv/bin/python scripts/mac/record_environment.py
.venv/bin/python scripts/mac/verify_data.py  # exits 1 for the eight documented SFT byte mismatches
.venv/bin/python scripts/mac/inspect_adapters.py
.venv/bin/python scripts/mac/audit_templates.py --tokenizer runs/med/adapter --out results/mac/templates-tokenizer
.venv/bin/python -m pytest tests/test_mac_harness.py -q
```
