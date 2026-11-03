"""G.3 — one LoRA step at sequence length 1,024. Run with `bf16` (plan) or `fp16` as argv[1]."""
import json, os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

prec = sys.argv[1] if len(sys.argv) > 1 else "bf16"
dtype = torch.bfloat16 if prec == "bf16" else torch.float16
name = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map="cuda")
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()
torch.cuda.reset_peak_memory_stats()
x = torch.randint(0, tok.vocab_size, (1, 1024), device="cuda")
t0 = time.time()
loss = model(input_ids=x, labels=x).loss
loss.backward()
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"[{prec}] loss {loss.item():.3f} | step {dt:.1f}s | peak GB: {peak:.2f} | finite: {torch.isfinite(loss).item()}")
os.makedirs("results/smoke", exist_ok=True)
json.dump({"test": "G.3", "precision": prec, "loss": loss.item(), "peak_gb": peak, "step_s": dt,
           "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
          open(f"results/smoke/g3_{prec}.json", "w", encoding="utf-8"), indent=1)
