"""IndicTrans2 en→hi drafts. Used for (a) the XSTest 150 sheet (I.6.2) and (b) the Alpaca harmless prompts for the
three-language calibration pairs (I.9). Drafts are post-edited by hand afterwards; `origin` records that.

    python scripts/data/translate_indictrans2.py xstest
    python scripts/data/translate_indictrans2.py calib
"""
import json, os, sys, torch
DEV = os.environ.get("IT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
from common import *
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# transformers>=5 moved PreTrainedTokenizerBase; IndicTransToolkit still imports it from tokenization_utils
import transformers.tokenization_utils as _tu
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
_tu.PreTrainedTokenizerBase = _PTB
from IndicTransToolkit.processor import IndicProcessor

job = sys.argv[1]
it_name = "ai4bharat/indictrans2-en-indic-1B"
tok = AutoTokenizer.from_pretrained(it_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(it_name, trust_remote_code=True, dtype=(torch.float16 if DEV == "cuda" else torch.float32)).to(DEV).eval()
ip = IndicProcessor(inference=True)

def translate(sents, bs=16):
    out = []
    for i in range(0, len(sents), bs):
        batch = ip.preprocess_batch(sents[i:i + bs], src_lang="eng_Latn", tgt_lang="hin_Deva")
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
        with torch.inference_mode():
            gen = model.generate(**enc, max_length=256, num_beams=5, num_return_sequences=1, use_cache=False)  # remote code breaks with the new Cache API
        out += ip.postprocess_batch(tok.batch_decode(gen, skip_special_tokens=True), lang="hin_Deva")
        print(f"{i + len(batch)}/{len(sents)}", flush=True)
    return out

if job == "xstest":
    sheet = read_jsonl(INTERIM / "xstest_en150_sheet.jsonl")
    hi = translate([r["english"] for r in sheet])
    for r, h in zip(sheet, hi):
        r["hindi_draft"] = h
    write_jsonl(INTERIM / "xstest_en150_sheet.jsonl", sheet)
    print("xstest drafts written")
elif job == "calib":
    rows = read_jsonl(INTERIM / "calib_harmless_en_for_hi.jsonl")
    hi = translate([r["text"] for r in rows])
    for r, h in zip(rows, hi):
        r["hindi_draft"] = h
    write_jsonl(INTERIM / "calib_harmless_hi_draft.jsonl", rows)
    print("calibration harmless Hindi drafts written")
