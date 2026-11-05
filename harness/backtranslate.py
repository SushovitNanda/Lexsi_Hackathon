"""Back-translate Hindi / Hinglish prompts and answers of a judged cell to English for WildGuard.

Used (a) for the G1 judge comparison on the hand-label set and (b) for every Hindi answer of every reported cell,
because WildGuard-on-back-translation is the Hindi primary judge (PREREG amendment 1, 5 Nov).

Runs on the SYSTEM Python (transformers 4.57, IndicTransToolkit; the venv's transformers 5 breaks the remote code):

    set PYTHONUTF8=1 && set IT_DEVICE=cuda && C:\\Users\\SUSHOVIT\\python.exe harness/backtranslate.py --cell <cell> [--cell ...] [--langs hi]

IndicTrans2 indic->en reads Devanagari (hin_Deva). Latin-script Hinglish has no IndicTrans2 source language, so:
  * Devanagari text (Hindi prompts/answers, Devanagari Hinglish answers)  -> translated   (bt_mode = "indictrans2")
  * Latin-script Hinglish text                                             -> passed through (bt_mode = "latin_passthrough")
  * mixed text: translated as hin_Deva (Latin words pass through the model)  (bt_mode = "indictrans2_mixed")
Writes results/judges/backtranslation_<cell>.jsonl (the hand-label cell keeps its original name
backtranslation_hand_labels.jsonl): prompt_id, lang, prompt_en, text_en, bt_prompt_mode, bt_text_mode. Resumable.
English items are copied through unchanged.
"""
import argparse, json, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
DEV = os.environ.get("IT_DEVICE", "cpu")
DEVA = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
NAME = "ai4bharat/indictrans2-indic-en-1B"


def out_path(cell):
    if cell == "hand_labels__orig__fp16":
        return ROOT / "results" / "judges" / "backtranslation_hand_labels.jsonl"
    return ROOT / "results" / "judges" / f"backtranslation_{cell}.jsonl"


def mode(t):
    d, l = len(DEVA.findall(t)), len(LATIN.findall(t))
    if d == 0: return "latin_passthrough"
    return "indictrans2" if d >= 0.8 * (d + l) else "indictrans2_mixed"


def split_sents(t, max_chars=400):
    """IndicTrans2 is trained on sentences: split on danda / newline / sentence punctuation, keep chunks short."""
    parts = [p.strip() for p in re.split(r"(?<=[।\.\?\!])\s+|\n+", t) if p.strip()]
    out = []
    for p in parts:
        while len(p) > max_chars:
            out.append(p[:max_chars]); p = p[max_chars:]
        out.append(p)
    return out or [t.strip() or "."]


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", help="default: hand_labels__orig__fp16")
    ap.add_argument("--langs", default="hi,hinglish", help="comma list of languages to back-translate (others skipped; en copied)")
    ap.add_argument("--batch", type=int, default=16 if DEV == "cuda" else 8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N items per cell")
    a = ap.parse_args()
    cells = a.cell or ["hand_labels__orig__fp16"]
    langs = set(a.langs.split(","))

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import transformers.tokenization_utils as _tu
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
    _tu.PreTrainedTokenizerBase = _PTB
    from IndicTransToolkit.processor import IndicProcessor
    tok = AutoTokenizer.from_pretrained(NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(NAME, trust_remote_code=True,
                                                  dtype=(torch.float16 if DEV == "cuda" else torch.float32)).to(DEV).eval()
    ip = IndicProcessor(inference=True)

    def translate(sents):
        res = []
        for i in range(0, len(sents), a.batch):
            batch = ip.preprocess_batch(sents[i:i + a.batch], src_lang="hin_Deva", tgt_lang="eng_Latn")
            enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
            with torch.inference_mode():
                # max_new_tokens caps a degenerate beam (one item ran 30+ min on CPU without it); 2 beams for speed
                gen = model.generate(**enc, max_new_tokens=96, num_beams=2, num_return_sequences=1, use_cache=False)
            res += ip.postprocess_batch(tok.batch_decode(gen, skip_special_tokens=True), lang="eng_Latn")
        return res

    def bt(text):
        m = mode(text)
        if m == "latin_passthrough":
            return text, m
        return " ".join(translate(split_sents(text))), m

    for cell in cells:
        src = ROOT / "results" / "private" / cell / "responses.jsonl"
        out = out_path(cell); out.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
        rows = [r for r in rows if r["lang"] == "en" or r["lang"] in langs]
        if a.limit: rows = rows[: a.limit]
        done = {json.loads(l)["prompt_id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
        todo = [r for r in rows if r["prompt_id"] not in done]
        print(f"[bt {cell}] {len(rows)} items, {len(todo)} to do, device {DEV}, langs {sorted(langs)}", flush=True)
        t0 = time.time()
        for k, r in enumerate(todo):
            if r["lang"] == "en":
                o = {"prompt_id": r["prompt_id"], "lang": "en", "prompt_en": r["prompt"], "text_en": r["text"],
                     "bt_prompt_mode": "english", "bt_text_mode": "english"}
            else:
                p_en, pm = bt(r["prompt"]); t_en, tm = bt(r["text"]) if r["text"].strip() else ("", "empty")
                o = {"prompt_id": r["prompt_id"], "lang": r["lang"], "prompt_en": p_en, "text_en": t_en,
                     "bt_prompt_mode": pm, "bt_text_mode": tm}
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
            if k % 20 == 0:
                print(f"[bt {cell}] {k + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        print(f"[bt {cell}] DONE in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
