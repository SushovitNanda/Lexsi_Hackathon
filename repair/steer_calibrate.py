"""Refusal direction per layer (CAA-style; SafeTune's compute_refusal_direction is the same construction):
  v_l = mean_h[ resid_l(harmful) ] - mean_h[ resid_l(harmless) ]   at the last prompt token, official template, FP16.

    python repair/steer_calibrate.py --pairs trilingual --model models/merged/med --out models/steer/med_trilingual.pt
    python repair/steer_calibrate.py --pairs en_only  ...                                  (RQ9-style, optional)

Saves {"layers": [...], "vectors": {layer: tensor(d_model)}, "norms": {...}, "n_harmful", "n_harmless", "model", "pairs"}.
Calibration pairs: Datasets/final/calibration/calib_<pairs>_{harmful,harmless}.jsonl (never test items).
"""
import argparse, json, os, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/merged/med"); ap.add_argument("--pairs", default="trilingual", choices=["trilingual", "en_only"])
    ap.add_argument("--layers", default="16,18,20,22,24,26,27,28,30,32,33,34"); ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="cuda").eval()
    cal = ROOT / "Datasets" / "final" / "calibration"
    H = [json.loads(l) for l in open(cal / f"calib_{a.pairs}_harmful.jsonl", encoding="utf-8") if l.strip()]
    S = [json.loads(l) for l in open(cal / f"calib_{a.pairs}_harmless.jsonl", encoding="utf-8") if l.strip()]

    def mean_resid(rows):
        acc = {l: torch.zeros(model.config.hidden_size, dtype=torch.float64) for l in layers}; n = 0
        for i in range(0, len(rows), a.bs):
            chunk = rows[i:i + a.bs]
            texts = [tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states              # hs[l+1] = output of decoder layer l
            for l in layers:
                acc[l] += hs[l + 1][:, -1, :].double().sum(0).cpu()
            n += len(chunk)
        return {l: (acc[l] / n) for l in layers}

    t0 = time.time()
    mh, ms = mean_resid(H), mean_resid(S)
    vec = {l: (mh[l] - ms[l]).float() for l in layers}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"layers": layers, "vectors": vec, "norms": {l: float(v.norm()) for l, v in vec.items()},
                "n_harmful": len(H), "n_harmless": len(S), "model": a.model, "pairs": a.pairs}, a.out)
    print(json.dumps({"out": a.out, "n_harmful": len(H), "n_harmless": len(S), "norms": {l: round(float(v.norm()), 2) for l, v in vec.items()},
                      "seconds": round(time.time() - t0)}, indent=1))


if __name__ == "__main__":
    main()
