# models/ — every model file and cache lives here (user rule, 3 Nov 2026)

| folder | contents |
|---|---|
| `hf_cache/` | the Hugging Face cache (`HF_HOME`): tiny-aya-global, tiny-aya-base, Qwen3Guard-Gen-4B, WildGuard, IndicTrans2, the three datasets, Detoxify/sentence-transformer caches |
| `runs/` | LoRA runs: `med/` and `ctrl/` (adapter + checkpoints + `train_summary.json`), later repairs and sweeps |
| `merged/` | merged full weights (FP16 safetensors) for the GGUF path and the ReSta/merging/projection arithmetic |

Set `HF_HOME` to this `hf_cache` folder (user env var + `.env`). Nothing here is pushed anywhere.
Note: this folder sits inside OneDrive; exclude it from sync (OneDrive → Settings → Sync and backup → Choose folders) or it will try to upload ~50 GB.
