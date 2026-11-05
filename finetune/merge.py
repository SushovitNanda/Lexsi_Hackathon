"""Phase 3 — merge a LoRA adapter into Tiny Aya Global on CPU in FP32, save FP16 safetensors + sha256 (Sushovit_Task §4.3).

    python finetune/merge.py models/runs/med/adapter models/merged/med
    python finetune/merge.py models/runs/ctrl/adapter models/merged/ctrl
Needs ~14 GB free RAM for the FP32 merge; falls back to BF16 on CPU if `--bf16` is given.
"""
import argparse, hashlib, json, os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("adapter"); ap.add_argument("out"); ap.add_argument("--bf16", action="store_true")
ap.add_argument("--base-model", default="CohereLabs/tiny-aya-global", help="Hub ID or pinned local snapshot")
ap.add_argument("--base-revision", help="Pinned Hub revision, recorded in merge_info.json")
a = ap.parse_args()
name = a.base_model
t0 = time.time()
dtype = torch.bfloat16 if a.bf16 else torch.float32
base = AutoModelForCausalLM.from_pretrained(name, revision=a.base_revision, dtype=dtype, device_map="cpu")
m = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
m = m.to(torch.float16)
os.makedirs(a.out, exist_ok=True)
m.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(name, revision=a.base_revision).save_pretrained(a.out)
hashes = {}
for f in sorted(os.listdir(a.out)):
    if f.endswith(".safetensors"):
        h = hashlib.sha256()
        with open(os.path.join(a.out, f), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        hashes[f] = h.hexdigest()
json.dump({"base": name, "base_revision": a.base_revision, "adapter": a.adapter, "merge_dtype": str(dtype), "saved_dtype": "float16",
           "sha256": hashes, "seconds": time.time() - t0},
          open(os.path.join(a.out, "merge_info.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(hashes, indent=1), f"\nmerged in {time.time()-t0:.0f}s -> {a.out}")
