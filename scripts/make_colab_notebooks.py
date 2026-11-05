"""Write the Colab notebooks into colab/. Run after scripts/make_drive_bundle.py.

    python scripts/make_colab_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
    ("orig__fp16__f16__tpl-official", "orig", "fp16"),
    ("orig__nf4__f16__tpl-official", "orig", "nf4"),
    ("med__fp16__f16__tpl-official", "med", "fp16"),
    ("med__nf4__f16__tpl-official", "med", "nf4"),
    ("ctrl__fp16__f16__tpl-official", "ctrl", "fp16"),
    ("ctrl__nf4__f16__tpl-official", "ctrl", "nf4"),
]
SETS = ["aya_en_test", "aya_hi_test", "aya_hinglish_test", "clinicsafe_hi_en", "clinicsafe_hi_hi", "clinicsafe_hi_hinglish",
        "xstest_en_test", "xstest_hi_test", "xstest_hinglish_test"]

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})

md("""# c1 — Reported drift cells on a Colab free T4 (Task 2, RQ1 / RQ2 / RQ3-baseline / RQ4)

**Before you start**
1. On the laptop: `python scripts/make_drive_bundle.py`, then upload the folder `drive_bundle/LexsiDrive` to **My Drive** (so it is `My Drive/LexsiDrive`). ~300 MB.
2. Runtime → Change runtime type → **T4 GPU**. Run the cells top to bottom.
3. Edit `RUN_CELLS` in step 4 to choose which cells this session runs. Everything is **resumable**: if the session dies, rerun all cells and it continues where it stopped.

**What it does:** pins the laptop's library versions → mounts Drive → rebuilds the merged medical/control models from the adapters and **checks every tensor hash against the laptop's merged files** (it stops on any mismatch) → generates each cell (1,612 prompts, FP16 greedy, 256 tokens, official template) → scores MedMCQA test-500 → prints a summary block.

**Judging is not done here.** Responses are written to `My Drive/LexsiDrive/outputs/`; copy that folder into the repo's `results/` afterwards (see the last cell). Qwen3Guard/WildGuard run locally so every cell shares one judge environment.

| Cell | Expected T4 time |
|---|---|
| `orig/med/ctrl` FP16 | ~35–50 min each |
| `orig/med/ctrl` NF4 | ~60–100 min each |

A 4-hour session fits about two FP16 + one NF4 cell. Suggested split: **session 1** = orig FP16, orig NF4, med FP16; **session 2** = med NF4, ctrl FP16, ctrl NF4. The first chunk prints the real speed; plan from it.
""")
code("""# 1. GPU check + the laptop's pinned versions (ENV.md). Colab's preinstalled torch is kept.
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q "transformers==5.22.1" "peft==0.21.2" "bitsandbytes==0.50.2" "accelerate==1.15.0" "huggingface_hub==1.33.0" "safetensors==0.8.0" "sentencepiece==0.2.2"
import torch, transformers, peft, bitsandbytes
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| transformers", transformers.__version__, "| peft", peft.__version__, "| bnb", bitsandbytes.__version__)
assert torch.cuda.is_available(), "No GPU: set Runtime -> Change runtime type -> T4 GPU"
""")
code("""# 2. Drive + code + prompts
import os, zipfile, json, time, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/LexsiDrive"
assert os.path.exists(f"{DRIVE}/code.zip"), "Upload drive_bundle/LexsiDrive to My Drive first"
WORK = "/content/lexsi"; os.makedirs(WORK, exist_ok=True)
for z in ("code.zip", "prompts.zip"):
    zipfile.ZipFile(f"{DRIVE}/{z}").extractall(WORK)
OUT = f"{DRIVE}/outputs"; os.makedirs(OUT, exist_ok=True)
os.environ["HF_HOME"] = "/content/hf"; os.environ["PYTHONUTF8"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print(json.load(open(f"{DRIVE}/bundle_info.json")))
""")
code("""# 3. Hugging Face READ token (typed here, never stored in the notebook) — Tiny Aya is gated
from getpass import getpass
from huggingface_hub import login
login(token=getpass("HF read token: "))
""")
code(f"""# 4. Choose this session's cells (edit this list)
ALL_CELLS = {json.dumps([c[0] for c in CELLS], indent=1)}
RUN_CELLS = ALL_CELLS[:3]      # session 1: orig FP16, orig NF4, med FP16   |   session 2: ALL_CELLS[3:]
SETS = {json.dumps(SETS)}
MODEL_OF = {{c: c.split("__")[0] for c in ALL_CELLS}}
WEIGHTS_OF = {{c: c.split("__")[1] for c in ALL_CELLS}}
print("this session:", RUN_CELLS)
""")
code("""# 5. Rebuild the merged models this session needs (streamed, fits Colab RAM) and verify every tensor hash
import glob
from safetensors import safe_open
import hashlib
need = sorted({MODEL_OF[c] for c in RUN_CELLS} - {"orig"})
MODEL_PATH = {"orig": "CohereLabs/tiny-aya-global"}
for m in need:
    out = f"/content/merged/{m}"
    if not os.path.exists(f"{out}/tensor_sha256.json"):
        r = subprocess.run([sys.executable, f"{WORK}/finetune/merge_stream.py", "--adapter", f"{DRIVE}/adapters/{m}", "--out", out],
                           capture_output=True, text=True)
        print(r.stdout[-600:], r.stderr[-1500:])
        assert r.returncode == 0, "merge failed"
    mine = json.load(open(f"{out}/tensor_sha256.json"))["tensor_sha256"]
    ref = json.load(open(f"{DRIVE}/manifests/{m}_tensor_sha256.json"))["tensor_sha256"]
    bad = [k for k in ref if mine.get(k) != ref[k]]
    print(f"{m}: {len(ref) - len(bad)}/{len(ref)} tensors identical to the laptop's merged model")
    assert not bad, f"HASH MISMATCH in {len(bad)} tensors (e.g. {bad[:3]}): stop and send this output back"
    MODEL_PATH[m] = out
print("models ready:", MODEL_PATH)
""")
code("""# 6. Generate each cell (resumable) + MedMCQA test-500 one-token accuracy
t_session = time.time()
for cell in RUN_CELLS:
    m, w = MODEL_OF[cell], WEIGHTS_OF[cell]
    print(f"\\n===== {cell}  (model={m}, weights={w}) =====", flush=True)
    cmd = [sys.executable, f"{WORK}/harness/run_cell.py", "--cell", cell, "--model", MODEL_PATH[m], "--weights", w,
           "--sets", *SETS, "--out", OUT, "--batch", "16"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=WORK)
    for line in p.stdout:
        if line.startswith("[") or "OOM" in line or "Error" in line or "Traceback" in line:
            print(line, end="", flush=True)
    assert p.wait() == 0, f"{cell} failed: rerun this cell to resume"
    mq = f"{OUT}/cells/{cell}/medmcqa_test500.json"
    if not os.path.exists(mq):
        cmd = [sys.executable, f"{WORK}/scripts/eval/medmcqa_score.py", "--model", MODEL_PATH[m],
               "--set", f"{WORK}/Datasets/final/medmcqa_test_500/prompts.jsonl", "--out", mq] + (["--nf4"] if w == "nf4" else [])
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=WORK)
        print([l for l in r.stdout.splitlines() if "accuracy" in l] or r.stderr[-800:])
    print(f"session elapsed {(time.time() - t_session) / 3600:.2f} h", flush=True)
""")
code("""# 7. Summary block: copy everything between the markers back into the chat
summary = {}
for cell in ALL_CELLS:
    rec = f"{OUT}/cells/{cell}/records.jsonl"
    info = f"{OUT}/cells/{cell}/run_info.json"
    mq = f"{OUT}/cells/{cell}/medmcqa_test500.json"
    s = {"generated": sum(1 for _ in open(rec, encoding="utf-8")) if os.path.exists(rec) else 0, "expected": 1612}
    if os.path.exists(info): s.update({k: v for k, v in json.load(open(info)).items() if k in ("gen_per_min", "tok_per_s", "peak_gb", "seconds", "engine")})
    if os.path.exists(mq): s["medmcqa_test500"] = round(json.load(open(mq))["accuracy"], 4)
    summary[cell] = s
print("=====C1_RESULTS_BEGIN====="); print(json.dumps(summary, indent=1)); print("=====C1_RESULTS_END=====")
print("\\nNext: download My Drive/LexsiDrive/outputs/ and copy its 'private' and 'cells' folders into the repo's results/ "
      "(results/private/<cell>/ and results/cells/<cell>/). Judging then runs locally (Task 2h).")
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
p = ROOT / "colab" / "c1_drift_cells.ipynb"
json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written", p)
