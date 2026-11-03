"""G.3c — the G.3/G.3b steps ran the model in eval mode (from_pretrained default), and transformers only applies
gradient checkpointing when `self.training` is True, so activations were kept for all 36 layers (~20 GB).
Repeat the step with model.train(), as Trainer does: A' plain full-logits (plan recipe), C' assistant-only logits,
D' QLoRA NF4 (fallback). Batch 1, seq 1,024, 40 % loss positions."""
import gc, json, os, sys, time, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

name = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(name)
L = 1024; torch.manual_seed(0)
x = torch.randint(0, tok.vocab_size, (1, L), device="cuda")
labels = x.clone(); labels[:, : int(L * 0.6)] = -100
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
res = {}

def assistant_only_loss(model, input_ids, labels):
    base = model.get_base_model()
    h = base.model(input_ids=input_ids, use_cache=False).last_hidden_state[:, :-1, :]
    y = labels[:, 1:]; m = y != -100
    logits = base.lm_head(h[m]).float()
    ls = getattr(base.config, "logit_scale", None)
    if ls: logits = logits * ls
    return F.cross_entropy(logits, y[m])

def measure(label, model, fn, steps=2):
    model.train()
    for i in range(steps):                       # second step = steady state
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); loss = fn(model); loss.backward(); torch.cuda.synchronize(); dt = time.time() - t0
        model.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    res[label] = {"loss": loss.item(), "peak_gb": peak, "step_s": dt, "finite": bool(torch.isfinite(loss)),
                  "training_mode": model.training}
    print(f"[{label}] loss {loss.item():.3f} | step {dt:.1f}s | peak GB {peak:.2f} | train={model.training}", flush=True)

def load(dtype, reentrant):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map="cuda")
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    m.enable_input_require_grads()
    return get_peft_model(m, lora)

which = sys.argv[1:] or ["A", "C", "D", "F"]
if "A" in which:
    m = load(torch.bfloat16, True); measure("A_train_bf16_full_logits", m, lambda m: m(input_ids=x, labels=labels).loss); del m
if "F" in which:
    m = load(torch.float16, True); measure("F_train_fp16_full_logits", m, lambda m: m(input_ids=x, labels=labels).loss); del m
if "C" in which:
    m = load(torch.bfloat16, False); measure("C_train_bf16_assistant_only", m, lambda m: assistant_only_loss(m, x, labels)); del m
if "D" in which:
    gc.collect(); torch.cuda.empty_cache()
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map="cuda")
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    m = get_peft_model(m, lora); measure("D_train_qlora_nf4_assistant_only", m, lambda m: assistant_only_loss(m, x, labels)); del m
os.makedirs("results/smoke", exist_ok=True)
json.dump(res, open("results/smoke/g3c.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
