"""G.3b — LoRA memory variants at seq 1,024 (the plain G.3 step peaked at 20.8 GB and spilled to system RAM).
Variants, each measured for peak VRAM and step time (forward + backward, batch 1, 1,024 tokens, ~40% of
positions carrying loss, as in the real SFT data):
  A  plain: full logits, reentrant checkpointing (= G.3)                      [reference]
  B  use_cache=False + non-reentrant checkpointing, full logits
  C  B + assistant-only logits (lm_head applied only at label positions)       [loss on assistant tokens only, exact]
  D  C on an NF4 base (QLoRA, FP16 compute)                                    [plan's fallback if memory is tight]
"""
import gc, json, os, sys, time, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

name = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(name)
L = 1024
torch.manual_seed(0)
x = torch.randint(0, tok.vocab_size, (1, L), device="cuda")
labels = x.clone(); labels[:, : int(L * 0.6)] = -100          # loss on the last 40% (assistant turn)
res = {}
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")

def assistant_only_loss(model, input_ids, labels):
    """Hidden states from the backbone, lm_head only where the (shifted) label is not -100."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    out = base.model(input_ids=input_ids, use_cache=False)
    h = out.last_hidden_state[:, :-1, :]                      # predict token t+1 from position t
    y = labels[:, 1:]
    m = y != -100
    hs = h[m]                                                 # (n_loss_tokens, d_model)
    logits = base.lm_head(hs).float()
    if getattr(base, "logit_scale", None) is not None:
        logits = logits * base.logit_scale
    elif getattr(base.config, "logit_scale", None):
        logits = logits * base.config.logit_scale
    return F.cross_entropy(logits, y[m])

def measure(label, model, fn):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    loss = fn(model); loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    res[label] = {"loss": loss.item(), "peak_gb": peak, "step_s": dt, "finite": bool(torch.isfinite(loss))}
    print(f"[{label}] loss {loss.item():.3f} | step {dt:.1f}s | peak GB {peak:.2f}", flush=True)
    model.zero_grad(set_to_none=True)

def load_bf16(reentrant):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    m.enable_input_require_grads()
    return get_peft_model(m, lora)

which = sys.argv[1:] or ["B", "C", "D"]
if "A" in which:
    m = load_bf16(True); measure("A_plain_reentrant", m, lambda m: m(input_ids=x, labels=labels).loss); del m
if "B" in which:
    m = load_bf16(False); measure("B_nonreentrant_full_logits", m, lambda m: m(input_ids=x, labels=labels).loss); del m
if "C" in which:
    m = load_bf16(False); measure("C_nonreentrant_assistant_only", m, lambda m: assistant_only_loss(m, x, labels)); del m
if "D" in which:
    gc.collect(); torch.cuda.empty_cache()
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map="cuda")
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True,
                                        gradient_checkpointing_kwargs={"use_reentrant": False})
    m = get_peft_model(m, lora)
    measure("D_qlora_nf4_assistant_only", m, lambda m: assistant_only_loss(m, x, labels)); del m
os.makedirs("results/smoke", exist_ok=True)
json.dump(res, open("results/smoke/g3b.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
