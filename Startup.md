# Startup — set up this repo on a new machine

Run from a terminal in order. Commands are bash (Linux, Kaggle, Git Bash on Windows); PowerShell equivalents are in the comments.
Needs: Python 3.12, git, an NVIDIA GPU with a CUDA 12.x driver, ~60 GB free disk, a Hugging Face token with access to the gated repos.

## Not in Git: shared over Google Drive

Upload these from the original machine and unzip them into the repo root on the new one (paths stay the same).
`results/` and `logs/` are small and come with the Git repo.

| Folder | Size | Contents | Upload? |
|---|---|---|---|
| `models/runs/` (without the `checkpoint-*` subfolders) | ~410 MB | LoRA adapters `med`, `ctrl`, `med_e2` + `train_summary.json` loss curves | **Yes** |
| `models/merged/` | 13 GB | merged FP16 medical and control weights | No — rebuilt from the adapters in ~30 s each (step 12) |
| `models/runs/*/checkpoint-*` | 1.5 GB | optimizer states from training | No — not needed |
| `models/hf_cache/` | 45 GB | base models and source datasets | No — re-downloaded (step 9) |
| `cache/` | <1 MB | CircuitKIT task cache | No — regenerated automatically |

```bash
# 1. Get the code.
git clone <REPO_URL> Lexsi_Hackathon
cd Lexsi_Hackathon

# 2. Python 3.12 virtual environment (SafeTune requires >= 3.12).
#    PowerShell: py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# 3. PyTorch with CUDA first, so no later package pulls in a CPU-only build.
#    Skip on Colab/Kaggle (torch is preinstalled there).
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 4. The three Lexsi kits, cloned at the exact commits used (unmodified upstream code), installed editable.
git clone https://github.com/Lexsi-Labs/CuratorKIT.git KITS/CuratorKIT && git -C KITS/CuratorKIT checkout dd9e835301fcddc74a37501d9e207a5a060d209f
git clone https://github.com/Lexsi-Labs/SafeTune.git   KITS/SafeTune   && git -C KITS/SafeTune   checkout 26f59e6d1bdcacafe5f2be3b8d982fb214369659
git clone https://github.com/Lexsi-Labs/CircuitKIT.git KITS/CircuitKIT && git -C KITS/CircuitKIT checkout e71d5e17fe6e2abfde47ec9a7fbaf31649601b78
pip install -e "KITS/CuratorKIT[hf,embedding,hygiene]" -e KITS/SafeTune -e KITS/CircuitKIT

# 5. Pinned HF stack (run after the kits so their dependencies cannot drift the pins).
pip install -r requirements.txt

# 6. Check CUDA survived and the kits import (circuitkit must be imported before transformer_lens).
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import circuitkit, safetune, curatorkit, auditkit; print('kits ok')"

# 7. UTF-8 mode (Hindi text breaks without it on Windows) and keep every model file inside the repo.
#    PowerShell: setx PYTHONUTF8 1 ; setx HF_HOME "$PWD\models\hf_cache"   (then open a new terminal)
export PYTHONUTF8=1
export HF_HOME="$PWD/models/hf_cache"

# 8. Hugging Face token (read access to Tiny Aya, WildGuard, IndicTrans2). .env is git-ignored.
#    PowerShell: $env:HF_TOKEN="hf_xxx"
export HF_TOKEN=hf_xxx
echo "HF_TOKEN=$HF_TOKEN" > .env
hf auth login --token "$HF_TOKEN"

# 9. Download the five models and three source datasets into models/hf_cache (~45 GB, the slow step).
python scripts/download_all.py

# 10. XSTest prompts CSV (only needed if you rebuild the data package; the final sets are already in Datasets/final/).
mkdir -p Datasets/raw/xstest
curl -L -o Datasets/raw/xstest/xstest_prompts.csv https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv

# 11. Smoke test: FP16 Tiny Aya Global with the official template (expects preamble present, ~6.8 GB VRAM).
python scripts/smoke/g1_fp16_generate.py

# 12. Fine-tuned models. Unzip the Google Drive copy of models/runs into the repo root.
#     Only if the Drive copy is unavailable, retrain the adapters (~40 min each on a 12 GB GPU):
#       export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#       python finetune/train_lora.py Datasets/final/sft_medical/train.jsonl models/runs/med  --precision fp16
#       python finetune/train_lora.py Datasets/final/sft_control/train.jsonl models/runs/ctrl --precision fp16
#     Then build the merged FP16 weights from the adapters (needed for the GGUF and repair work; ~14 GB RAM, CPU only).
python finetune/merge.py models/runs/med/adapter  models/merged/med
python finetune/merge.py models/runs/ctrl/adapter models/merged/ctrl

# 13. Check the medical model reproduces its MedMCQA dev accuracy (reference: 44.33 % merged FP16, 42.67 % original).
python scripts/eval/medmcqa_score.py --model models/merged/med --set Datasets/final/medmcqa_dev_300/prompts.jsonl
```

On a Colab/Kaggle T4 instead: open `colab/colab_first_cell_template.ipynb` and run it top to bottom; it pins the same versions and checks parity against the laptop numbers.
