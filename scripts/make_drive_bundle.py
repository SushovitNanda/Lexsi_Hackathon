"""Build the folder you upload to Google Drive for the Colab notebooks (nothing is pushed to any Hub).

    python scripts/make_drive_bundle.py            -> drive_bundle/LexsiDrive/  (upload the whole LexsiDrive folder to My Drive)

LexsiDrive/
  code.zip       harness/, finetune/merge_stream.py, scripts/eval/medmcqa_score.py
  prompts.zip    the Datasets/final sets the drift cells read (prompts only, no completions)
  adapters/med, adapters/ctrl     LoRA adapters (~121 MB each) — Colab rebuilds the merged models from these
  manifests/med_tensor_sha256.json, manifests/ctrl_tensor_sha256.json   per-tensor hashes of the laptop's merged models
  outputs/       (empty) the notebooks write responses and records here, resumable
"""
import json, shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "drive_bundle" / "LexsiDrive"
CELL_SETS = ["aya_en_test", "aya_hi_test", "aya_hinglish_test",
             "clinicsafe_hi_en", "clinicsafe_hi_hi", "clinicsafe_hi_hinglish",
             "xstest_en_test", "xstest_hi_test", "xstest_hinglish_test", "medmcqa_test_500"]

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "outputs").mkdir(parents=True); (OUT / "manifests").mkdir()

with zipfile.ZipFile(OUT / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted((ROOT / "harness").glob("*.py")):
        z.write(p, p.relative_to(ROOT))
    for rel in ("finetune/merge_stream.py", "scripts/eval/medmcqa_score.py", "requirements.txt"):
        z.write(ROOT / rel, rel)
with zipfile.ZipFile(OUT / "prompts.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for s in CELL_SETS:
        for f in ("prompts.jsonl", "manifest.json", "checksums.txt"):
            p = ROOT / "Datasets" / "final" / s / f
            if p.exists():
                z.write(p, p.relative_to(ROOT))
for m in ("med", "ctrl"):
    dst = OUT / "adapters" / m; dst.mkdir(parents=True)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(ROOT / "models" / "runs" / m / "adapter" / f, dst / f)
    shutil.copy(ROOT / "models" / "merged" / m / "tensor_sha256.json", OUT / "manifests" / f"{m}_tensor_sha256.json")
n = {s: sum(1 for _ in open(ROOT / "Datasets" / "final" / s / "prompts.jsonl", encoding="utf-8")) for s in CELL_SETS}
json.dump({"cell_sets": n, "generations_per_cell": sum(v for k, v in n.items() if k != "medmcqa_test_500")},
          open(OUT / "bundle_info.json", "w"), indent=1)
size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
print(f"bundle at {OUT}  ({size / 1e6:.0f} MB)  per-cell generations: {sum(v for k, v in n.items() if k != 'medmcqa_test_500')}")
