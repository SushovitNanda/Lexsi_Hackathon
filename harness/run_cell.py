"""Generate one evaluation cell: resumable, batched, FP16 greedy, official chat template.

    python harness/run_cell.py --cell probe_dev__orig__fp16 --model CohereLabs/tiny-aya-global \
        --sets aya_en_dev aya_hi_dev aya_hinglish_dev xstest_en_dev
    python harness/run_cell.py --cell med__nf4 --model models/merged/med --weights nf4 --sets ...
    python harness/run_cell.py --cell hand_labels__orig__fp16 --model CohereLabs/tiny-aya-global \
        --ids Datasets/final/hand_labels/sample_to_label.jsonl

Writes
  results/private/<cell>/responses.jsonl   full text (git-ignored; never published)
  results/cells/<cell>/records.jsonl       the same records WITHOUT text (labels, hashes, metadata)
  results/cells/<cell>/run_info.json       settings, throughput, timing
Finished prompt_ids are skipped on restart; records are flushed every --flush prompts.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import load_sets, load_ids, ROOT

DEVA = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def answer_script(text):
    d, l = len(DEVA.findall(text)), len(LATIN.findall(text))
    if d == 0 and l == 0:
        return "none"
    if d >= 0.8 * (d + l):
        return "devanagari"
    if l >= 0.8 * (d + l):
        return "latin"
    return "mixed"


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def render(tok, text, template):
    if template == "official":
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    if template == "none":   # turn tokens only, no System Preamble
        return (f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"
                f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")
    raise ValueError(template)


def load_model(path, weights, adapter=None):
    if weights == "nf4":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
        m = AutoModelForCausalLM.from_pretrained(path, quantization_config=q, device_map="cuda")
    else:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--adapter"); ap.add_argument("--weights", default="fp16", choices=["fp16", "nf4"])
    ap.add_argument("--template", default="official", choices=["official", "none"])
    ap.add_argument("--sets", nargs="*"); ap.add_argument("--ids")
    ap.add_argument("--max_new_tokens", type=int, default=256); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--flush", type=int, default=100); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results"))
    sys.path.insert(0, str(ROOT / "repair")); from resta import add_repair_args, load_patched
    add_repair_args(ap)
    a = ap.parse_args()
    torch.manual_seed(13)

    rows = load_ids(a.ids) if a.ids else load_sets(a.sets)
    if a.limit:
        rows = rows[: a.limit]
    priv = Path(a.out) / "private" / a.cell; pub = Path(a.out) / "cells" / a.cell
    priv.mkdir(parents=True, exist_ok=True); pub.mkdir(parents=True, exist_ok=True)
    resp_p, rec_p = priv / "responses.jsonl", pub / "records.jsonl"
    done = set()
    if resp_p.exists():
        done = {json.loads(l)["prompt_id"] for l in open(resp_p, encoding="utf-8") if l.strip()}
    todo = [r for r in rows if r["prompt_id"] not in done]
    print(f"[{a.cell}] {len(rows)} prompts, {len(done)} already done, {len(todo)} to go", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if a.resta is not None or a.steer:
        model, rep_info = load_patched(a.model, a.weights, a.adapter, a.resta, a.resta_tensors, a.steer, a.steer_layer, a.steer_strength)
    else:
        model, rep_info = load_model(a.model, a.weights, a.adapter), {}
    rendered = {r["prompt_id"]: render(tok, r["text"], a.template) for r in todo}
    lens = {k: len(tok(v, add_special_tokens=False).input_ids) for k, v in rendered.items()}
    todo.sort(key=lambda r: -lens[r["prompt_id"]])          # longest first: early OOM, less padding
    meta = {"model": a.model, "adapter": a.adapter, "weights": a.weights, "template": a.template,
            "engine": f"transformers {__import__('transformers').__version__}", "temperature": 0.0, "seed": 13,
            "max_tokens": a.max_new_tokens, "harness": git_sha(), **rep_info}
    buf_r, buf_p, n_done, t_start, gen_tokens = [], [], 0, time.time(), 0
    bs = a.batch
    i = 0
    while i < len(todo):
        chunk = todo[i:i + bs]
        texts = [rendered[r["prompt_id"]] for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        t0 = time.time()
        try:
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); bs = max(1, bs // 2); print(f"OOM -> batch {bs}", flush=True); continue
        dt = (time.time() - t0) * 1000 / len(chunk)
        for r, row in zip(chunk, out):
            new = row[enc.input_ids.shape[1]:]
            text = tok.decode(new, skip_special_tokens=True)
            ntok = int((new != tok.pad_token_id).sum().item())
            gen_tokens += ntok
            rec = {"prompt_id": r["prompt_id"], "cell": a.cell, "lang": r["lang"], "set": r["set"], "set_name": r["set_name"],
                   "category": r.get("category"), "twin_id": r.get("twin_id"),
                   "template_sha": hashlib.sha256(rendered[r["prompt_id"]].encode("utf-8")).hexdigest()[:16],
                   "n_tokens": ntok, "latency_ms": round(dt), "answer_script": answer_script(text), **meta}
            buf_r.append(rec); buf_p.append({**rec, "prompt": r["text"], "text": text})
        i += len(chunk); n_done += len(chunk)
        if len(buf_r) >= a.flush or i >= len(todo):
            with open(resp_p, "a", encoding="utf-8") as f:
                for x in buf_p: f.write(json.dumps(x, ensure_ascii=False) + "\n")
            with open(rec_p, "a", encoding="utf-8") as f:
                for x in buf_r: f.write(json.dumps(x, ensure_ascii=False) + "\n")
            buf_r, buf_p = [], []
            el = time.time() - t_start
            print(f"[{a.cell}] {len(done) + n_done}/{len(rows)} | {n_done / el * 60:.1f} gen/min | "
                  f"{gen_tokens / el:.0f} tok/s | peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    el = time.time() - t_start
    json.dump({**meta, "cell": a.cell, "n_prompts": len(rows), "generated_this_run": n_done, "seconds": round(el),
               "gen_per_min": round(n_done / el * 60, 1), "tok_per_s": round(gen_tokens / el),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "final_batch": bs},
              open(pub / "run_info.json", "w", encoding="utf-8"), indent=1)
    print(f"[{a.cell}] DONE in {el / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
