"""Training-free repairs for the harness (Plan v3 RQ4; Tasks/reference_notes.md §7).

ReSta (SafeTune's recover):  theta_rep = theta_med + alpha * (theta_Global - theta_Base)
  * apply_resta_inplace(model, alpha, only)  patches a loaded FP16 HF model tensor by tensor (dev sweeps: no disk copies)
  * build_resta(med_dir, out_dir, alpha, only) streams the same arithmetic to FP16 safetensors + per-tensor sha256
    (the reported artifact; FP32 arithmetic, FP16 save, same shard layout as finetune/merge_stream.py)
  `only` = set of HF parameter names (circuit-restricted ReSta); None = whole model.
  Verified against SafeTune apply_resta on sampled tensors by repair/check_resta.py.

Steering (baseline): add strength * v_l to the residual stream after decoder layer l (CAA-style direction from
repair/steer_calibrate.py). add_steer_hook(model, vec, layer, strength).

load_patched(path, weights, adapter, resta, resta_tensors, steer, steer_layer, steer_strength) is the single loader
used by harness/run_cell.py and scripts/eval/medmcqa_score.py.
"""
import glob, hashlib, json, os, shutil
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

GLOBAL, BASE = "CohereLabs/tiny-aya-global", "CohereLabs/tiny-aya-base"
_PAT = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"]


def _dir(name):
    return name if os.path.isdir(name) else snapshot_download(name, allow_patterns=_PAT)


def _index(d):
    p = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(p):
        return {k: os.path.join(d, f) for k, f in json.load(open(p))["weight_map"].items()}
    f = os.path.join(d, "model.safetensors")
    with safe_open(f, "pt") as h:
        return {k: f for k in h.keys()}


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def load_tensor_list(path):
    """JSON list of HF parameter names, or {"tensors": [...]} (locate/cresta_tensors_v2.json style)."""
    j = json.load(open(path, encoding="utf-8"))
    names = j["tensors"] if isinstance(j, dict) else j
    return set(names)


class Delta:
    """theta_Global - theta_Base, one tensor at a time, from the local HF cache."""
    def __init__(self):
        self.g, self.b = _index(_dir(GLOBAL)), _index(_dir(BASE))
        assert set(self.g) == set(self.b), "Global/Base tensor names differ"
        self._open = {}

    def _h(self, f):
        if f not in self._open: self._open[f] = safe_open(f, "pt")
        return self._open[f]

    def get(self, name, device="cpu"):
        g = self._h(self.g[name]).get_tensor(name).to(device).float()
        b = self._h(self.b[name]).get_tensor(name).to(device).float()
        return g - b

    def names(self):
        return list(self.g)


@torch.no_grad()
def apply_resta_inplace(model, alpha, only=None):
    """model: HF FP16 model on cuda. Adds alpha * delta to every (or `only`) 2-D weight it finds by name."""
    D = Delta(); sd = dict(model.named_parameters()); n = 0
    for name in D.names():
        if name not in sd or (only is not None and name not in only): continue
        p = sd[name]
        p.data = (p.data.float() + alpha * D.get(name, p.device)).to(p.dtype)
        n += 1
    # tied lm_head follows embed_tokens automatically
    return n


def build_resta(med_dir, out_dir, alpha, only=None, tag=None):
    D = Delta(); med = _index(med_dir)
    shards = sorted(set(med.values())); os.makedirs(out_dir, exist_ok=True)
    shas, n_patched, new_map = {}, 0, {}
    for si, shard in enumerate(shards):
        out = {}
        with safe_open(shard, "pt") as h:
            for k in h.keys():
                w = h.get_tensor(k).float()
                if only is None or k in only:
                    w = w + alpha * D.get(k); n_patched += 1
                t = w.half(); out[k] = t; shas[k] = tensor_sha(t)
        name = f"model-{si + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(out, os.path.join(out_dir, name), metadata={"format": "pt"})
        for k in out: new_map[k] = name
        del out
    json.dump({"metadata": {}, "weight_map": new_map}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)
    for f in glob.glob(os.path.join(med_dir, "*.json")) + glob.glob(os.path.join(med_dir, "*.jinja")):
        if not f.endswith("model.safetensors.index.json") and not f.endswith("tensor_sha256.json") and not f.endswith("merge_info.json"):
            shutil.copy(f, out_dir)
    json.dump({"method": "resta" if only is None else "cresta", "alpha": alpha, "tag": tag, "medical": med_dir,
               "global": GLOBAL, "base": BASE, "patched_tensors": n_patched, "restricted_to": sorted(only) if only else None,
               "tensor_sha256": shas}, open(os.path.join(out_dir, "tensor_sha256.json"), "w"), indent=1)
    return n_patched, shas


def add_steer_hook(model, vec, layer, strength):
    """vec: 1-D tensor (d_model). Added to the residual stream output of decoder layer `layer` at every position."""
    v = (vec.float() * strength)
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + v.to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return model.model.layers[layer].register_forward_hook(hook)


def load_patched(path, weights="fp16", adapter=None, resta=None, resta_tensors=None, steer=None, steer_layer=None, steer_strength=0.0):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if weights == "nf4":
        assert resta is None, "ReSta at load time needs FP16 weights; build the model with build_resta for NF4"
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        m = AutoModelForCausalLM.from_pretrained(path, quantization_config=q, device_map="cuda")
    else:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    info = {}
    if resta is not None:
        only = load_tensor_list(resta_tensors) if resta_tensors else None
        info["resta_alpha"] = resta; info["resta_tensors"] = resta_tensors
        info["resta_patched"] = apply_resta_inplace(m, resta, only)
    if steer:
        vecs = torch.load(steer, map_location="cpu")
        v = vecs["vectors"][int(steer_layer)] if isinstance(vecs, dict) and "vectors" in vecs else vecs[int(steer_layer)]
        add_steer_hook(m, v, int(steer_layer), float(steer_strength))
        info.update({"steer": steer, "steer_layer": int(steer_layer), "steer_strength": float(steer_strength)})
    return m.eval(), info


def add_repair_args(ap):
    ap.add_argument("--resta", type=float, help="ReSta alpha applied at load time (FP16 only)")
    ap.add_argument("--resta_tensors", help="JSON list of HF tensor names: circuit-restricted ReSta")
    ap.add_argument("--steer", help="steering vectors .pt from repair/steer_calibrate.py")
    ap.add_argument("--steer_layer", type=int); ap.add_argument("--steer_strength", type=float, default=0.0)
