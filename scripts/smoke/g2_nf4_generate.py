"""G.2 — NF4 load (the T4 path), FP16 compute, generate."""
import json, os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

name = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(name)
nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
t0 = time.time()
model4 = AutoModelForCausalLM.from_pretrained(name, quantization_config=nf4, device_map="cuda")
print(f"NF4 loaded in {time.time()-t0:.0f}s; footprint GB: {model4.get_memory_footprint()/1e9:.2f}")
msgs = [{"role": "user", "content": "BP ki tablet ke saath paracetamol lena safe hai kya?"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to("cuda")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
out = model4.generate(**ids, max_new_tokens=128, do_sample=False)
dt = time.time() - t0
ans = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
n_new = out.shape[1] - ids.input_ids.shape[1]
print("ANSWER:\n", ans)
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"new tokens {n_new} in {dt:.1f}s = {n_new/dt:.1f} tok/s | peak GB: {peak:.2f}")
os.makedirs("results/smoke", exist_ok=True)
json.dump({"test": "G.2", "quant": "nf4", "compute_dtype": "float16", "peak_gb": peak,
           "tok_per_s": n_new / dt, "answer": ans},
          open("results/smoke/g2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
