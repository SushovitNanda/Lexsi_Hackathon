"""Task 1 F: download every model and dataset once into HF_HOME (C:\hf).
Token comes from the environment (HF_TOKEN); never printed."""
import os, sys, time
from huggingface_hub import snapshot_download, login

tok = os.environ.get("HF_TOKEN")
assert tok, "HF_TOKEN missing"
login(token=tok, add_to_git_credential=False)

MODELS = [
    "CohereLabs/tiny-aya-global",          # main model (~6.7 GB)
    "CohereLabs/tiny-aya-base",            # ReSta base: patch = alpha * (Global - Base)
    "Qwen/Qwen3Guard-Gen-4B",              # primary judge
    "ai4bharat/indictrans2-en-indic-1B",   # Hindi drafts
    "allenai/wildguard",                   # second judge (~14.5 GB)
]
DATASETS = [
    "openlifescienceai/medmcqa",
    "yahma/alpaca-cleaned",
    "CohereForAI/aya_redteaming",
]
sel = sys.argv[1:] or ["models", "datasets"]
if "datasets" in sel:
    for d in DATASETS:
        t = time.time(); print("downloading dataset", d, flush=True)
        p = snapshot_download(d, repo_type="dataset")
        print("  ->", p, f"{time.time()-t:.0f}s", flush=True)
if "models" in sel:
    for m in MODELS:
        t = time.time(); print("downloading model", m, flush=True)
        p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.md", "*.jinja", "*.tiktoken"])
        print("  ->", p, f"{time.time()-t:.0f}s", flush=True)
print("ALL DOWNLOADS DONE", flush=True)
