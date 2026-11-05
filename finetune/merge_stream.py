"""Streaming LoRA merge: W_merged = fp16( fp32(W_base) + (alpha/r) * fp32(B) @ fp32(A) ), one tensor at a time.

Same arithmetic as finetune/merge.py (PEFT merge_and_unload on an FP32 base, saved FP16) but holds one tensor in RAM,
so it runs inside Colab's 12.7 GB. Writes FP16 safetensors (same shard layout as merge.py) + config/tokenizer, and a
per-tensor sha256 manifest so two machines can compare their merged weights.

    python finetune/merge_stream.py --adapter models/runs/med/adapter --out models/merged/med_stream
    python finetune/merge_stream.py --adapter ... --out ... --compare models/merged/med     # local equivalence check
"""
import argparse, glob, hashlib, json, os, shutil
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "CohereLabs/tiny-aya-global"


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE); ap.add_argument("--compare")
    a = ap.parse_args()
    base_dir = a.base if os.path.isdir(a.base) else snapshot_download(a.base, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
    cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    assert not cfg.get("use_rslora") and not cfg.get("use_dora")
    lora = {}
    with safe_open(os.path.join(a.adapter, "adapter_model.safetensors"), "pt") as h:
        for k in h.keys():
            lora[k] = h.get_tensor(k)
    # map base weight name -> (A, B)
    pairs = {}
    for k in lora:
        if k.endswith("lora_A.weight"):
            base_name = k.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
            pairs[base_name] = (lora[k], lora[k.replace("lora_A", "lora_B")])
    idx = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))["weight_map"]
    os.makedirs(a.out, exist_ok=True)
    shards = sorted(set(idx.values()))
    shas, merged_n = {}, 0
    for si, shard in enumerate(shards):
        out = {}
        with safe_open(os.path.join(base_dir, shard), "pt") as h:
            for k in h.keys():
                w = h.get_tensor(k).float()
                if k in pairs:
                    A, B = pairs[k]
                    w = w + (B.float() @ A.float()) * scale
                    merged_n += 1
                t = w.half()
                out[k] = t; shas[k] = tensor_sha(t)
        name = f"model-{si + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(out, os.path.join(a.out, name), metadata={"format": "pt"})
        print(f"shard {name}: {len(out)} tensors", flush=True)
        del out
    assert merged_n == len(pairs), (merged_n, len(pairs))
    # index + config/tokenizer from the base (same files merge.py writes)
    new_idx = {"metadata": {}, "weight_map": {}}
    for si, shard in enumerate(shards):
        name = f"model-{si + 1:05d}-of-{len(shards):05d}.safetensors"
        for k, v in idx.items():
            if v == shard: new_idx["weight_map"][k] = name
    json.dump(new_idx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    for f in glob.glob(os.path.join(base_dir, "*.json")) + glob.glob(os.path.join(base_dir, "*.jinja")):
        if not f.endswith("model.safetensors.index.json"):
            shutil.copy(f, a.out)
    c = json.load(open(os.path.join(a.out, "config.json"))); c["torch_dtype"] = "float16"; c["dtype"] = "float16"
    json.dump(c, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump({"adapter": a.adapter, "base": a.base, "scale": scale, "merged_tensors": merged_n, "tensor_sha256": shas},
              open(os.path.join(a.out, "tensor_sha256.json"), "w"), indent=1)
    print(f"merged {merged_n} LoRA targets into {len(shas)} tensors -> {a.out}")
    if a.compare:
        ref = {}
        for f in glob.glob(os.path.join(a.compare, "*.safetensors")):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    ref[k] = h.get_tensor(k)
        diff = [k for k in shas if k in ref and tensor_sha(ref[k]) != shas[k]]
        maxd = 0.0
        for k in diff[:40]:
            with safe_open(os.path.join(a.out, new_idx["weight_map"][k]), "pt") as h:
                maxd = max(maxd, (h.get_tensor(k).float() - ref[k].float()).abs().max().item())
        missing = [k for k in shas if k not in ref]
        print(f"compare vs {a.compare}: {len(shas) - len(diff) - len(missing)}/{len(shas)} tensors bit-identical, "
              f"{len(diff)} differ (max |diff| {maxd:.2e}), {len(missing)} not in reference")


if __name__ == "__main__":
    main()
