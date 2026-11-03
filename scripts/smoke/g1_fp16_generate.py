"""G.1 — load Tiny Aya Global in FP16, render the official template, generate (Hinglish)."""
import json, os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(name)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda")
print(f"loaded in {time.time()-t0:.0f}s; config: {model.config.model_type}, layers={model.config.num_hidden_layers}")

msgs = [{"role": "user", "content": "BP ki tablet ke saath paracetamol lena safe hai kya?"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
preamble = "contextual safety mode" in text
n_prompt = len(tok(text).input_ids)
n_pre = len(tok(tok.apply_chat_template([{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True)).input_ids)
print("preamble present:", preamble)
print("prompt tokens:", n_prompt, "| empty-user template tokens (~preamble):", n_pre)

ids = tok(text, return_tensors="pt").to("cuda")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
out = model.generate(**ids, max_new_tokens=128, do_sample=False)
dt = time.time() - t0
ans = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
n_new = out.shape[1] - ids.input_ids.shape[1]
print("ANSWER:\n", ans)
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"new tokens {n_new} in {dt:.1f}s = {n_new/dt:.1f} tok/s | peak GB: {peak:.2f}")
os.makedirs("results/smoke", exist_ok=True)
json.dump({"test": "G.1", "dtype": "float16", "preamble_present": preamble, "prompt_tokens": n_prompt,
           "preamble_tokens_est": n_pre, "peak_gb": peak, "tok_per_s": n_new / dt, "answer": ans,
           "rendered_prompt": text},
          open("results/smoke/g1.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
