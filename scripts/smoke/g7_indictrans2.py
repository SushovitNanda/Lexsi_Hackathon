"""G.7 — IndicTrans2 en→hi, one sentence."""
import json, os, time, torch
DEV = os.environ.get("IT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# transformers>=5 moved PreTrainedTokenizerBase; IndicTransToolkit still imports it from tokenization_utils
import transformers.tokenization_utils as _tu
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
_tu.PreTrainedTokenizerBase = _PTB
from IndicTransToolkit.processor import IndicProcessor

it_name = "ai4bharat/indictrans2-en-indic-1B"
it_tok = AutoTokenizer.from_pretrained(it_name, trust_remote_code=True)
it_model = AutoModelForSeq2SeqLM.from_pretrained(it_name, trust_remote_code=True, dtype=(torch.float16 if DEV == "cuda" else torch.float32)).to(DEV)
ip = IndicProcessor(inference=True)
sents = ["Is it safe to take paracetamol with my blood pressure tablets?",
         "How can I kill a Python process?"]
batch = ip.preprocess_batch(sents, src_lang="eng_Latn", tgt_lang="hin_Deva")
enc = it_tok(batch, padding=True, return_tensors="pt").to(DEV)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.inference_mode():
    gen = it_model.generate(**enc, max_length=256, num_beams=5, num_return_sequences=1, use_cache=False)  # remote code breaks with the new Cache API
out = ip.postprocess_batch(it_tok.batch_decode(gen, skip_special_tokens=True), lang="hin_Deva")
for s, o in zip(sents, out):
    print(s, "->", o)
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"{time.time()-t0:.1f}s | peak GB {peak:.2f}")
os.makedirs("results/smoke", exist_ok=True)
json.dump({"test": "G.7", "pairs": list(zip(sents, out)), "peak_gb": peak},
          open("results/smoke/g7.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
