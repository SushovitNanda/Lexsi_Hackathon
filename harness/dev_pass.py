"""One repair setting on the fixed dev pass (PREREG §6): generate -> judge -> MedMCQA dev-1,000 -> one row in
results/repair_dev/sweep.jsonl (ASR per language strict/primary, ORR general+clinical per language, MedMCQA with
McNemar vs the medical dev baseline, deltas vs the original / medical dev baselines).

    python harness/dev_pass.py --name resta_a0.25 --model models/merged/med --resta 0.25
    python harness/dev_pass.py --name cresta_a0.25_top10 --model models/merged/med --resta 0.25 --resta_tensors locate/cresta_tensors_v2_top10.json
    python harness/dev_pass.py --name steer_L27_s4 --model models/merged/med --steer models/steer/med_trilingual.pt --steer_layer 27 --steer_strength 4
    python harness/dev_pass.py --name baseline_med --model models/merged/med --cell dev_pass__med__fp16      # existing cells: metrics only

Baselines: dev_pass__orig__fp16 (ORR reference) and dev_pass__med__fp16 (MedMCQA + ASR reference), produced by chain A.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from stats import load_labels, metric_vec, boot_ci, mcnemar_from_preds, HARM, BENIGN
P = sys.executable
LANGS = ["en", "hi", "hinglish"]
DEV_IDS = ROOT / "Datasets" / "final" / "dev_pass_ids.jsonl"
MQD = ROOT / "Datasets" / "final" / "medmcqa_dev_1000" / "prompts.jsonl"


SYS_PY = os.environ.get("LEXSI_SYS_PY", r"C:\Users\SUSHOVIT\python.exe")   # system Python 3.13 (transformers 4.57, IndicTrans2)


def sh(cmd, log, env=None):
    with open(log, "a", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, text=True, env=env)
    assert r.returncode == 0, f"failed ({r.returncode}): {' '.join(map(str, cmd))} -> see {log}"


def metrics(cell):
    lab = load_labels(cell); out = {}
    for lang in LANGS + [None]:
        h = [i for i, r in lab.items() if r["set"] in HARM and (lang is None or r["lang"] == lang)]
        v = metric_vec(lab, h, "asr"); out[f"asr|{lang or 'all'}"] = round(float(v.mean()) * 100, 2)
        out[f"asr_sens|{lang or 'all'}"] = round(float(metric_vec(lab, h, "asr", True).mean()) * 100, 2)
        for bt, sets in (("general", {"benign_general"}), ("clinical", {"benign_clinical"})):
            b = [i for i, r in lab.items() if r["set"] in sets and (lang is None or r["lang"] == lang)]
            out[f"orr_{bt}|{lang or 'all'}"] = round(float(metric_vec(lab, b, "over_refusal").mean()) * 100, 2)
    out["unsure"] = sum(1 for r in lab.values() if r["safety"] == "unsure" or r["refusal"] == "unsure")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--model", default="models/merged/med"); ap.add_argument("--cell")
    ap.add_argument("--weights", default="fp16"); ap.add_argument("--adapter")
    ap.add_argument("--resta", type=float); ap.add_argument("--resta_tensors"); ap.add_argument("--steer"); ap.add_argument("--steer_layer", type=int)
    ap.add_argument("--steer_strength", type=float, default=0.0)
    ap.add_argument("--ref_orig", default="dev_pass__orig__fp16"); ap.add_argument("--ref_med", default="dev_pass__med__fp16")
    ap.add_argument("--sweep", default=str(ROOT / "results" / "repair_dev" / "sweep.jsonl"))
    a = ap.parse_args()
    cell = a.cell or f"dev_pass__{a.name}"
    log = ROOT / "logs" / f"t2_devpass_{a.name}.log"; (ROOT / "results" / "repair_dev").mkdir(parents=True, exist_ok=True)
    opts = []
    if a.adapter: opts += ["--adapter", a.adapter]
    if a.resta is not None: opts += ["--resta", str(a.resta)]
    if a.resta_tensors: opts += ["--resta_tensors", a.resta_tensors]
    if a.steer: opts += ["--steer", a.steer, "--steer_layer", str(a.steer_layer), "--steer_strength", str(a.steer_strength)]
    t0 = time.time()
    if not (ROOT / "results" / "cells" / cell / "labels_qwen3guard.jsonl").exists():
        sh([P, "harness/run_cell.py", "--cell", cell, "--model", a.model, "--weights", a.weights, "--ids", str(DEV_IDS)] + opts, log)
        sh([P, "harness/judge.py", "--cell", cell], log)
    # Hindi primary judge = WildGuard on IndicTrans2 back-translation (PREREG amendment 1): back-translate on the
    # system Python (IndicTrans2 needs transformers 4.x), then WildGuard 8-bit; both resumable.
    import stats
    wg_langs = sorted(l for l, j in stats.PRIMARY.items() if j == "wildguard")
    if wg_langs and not (ROOT / "results" / "cells" / cell / "labels_wildguard.jsonl").exists():
        env = dict(os.environ, IT_DEVICE="cuda", PYTHONUTF8="1")
        sh([SYS_PY, "harness/backtranslate.py", "--cell", cell, "--langs", ",".join(wg_langs)], log, env)
        sh([P, "harness/judge_wildguard.py", "--cell", cell, "--langs", ",".join(wg_langs)], log)
    mq = ROOT / "results" / "cells" / cell / "medmcqa_dev1000.json"
    if not mq.exists():
        sh([P, "scripts/eval/medmcqa_score.py", "--model", a.model, "--set", str(MQD), "--out", str(mq)] + (["--nf4"] if a.weights == "nf4" else []) + opts, log)
    row = {"name": a.name, "cell": cell, "model": a.model, "weights": a.weights, "resta": a.resta, "resta_tensors": a.resta_tensors,
           "steer": a.steer, "steer_layer": a.steer_layer, "steer_strength": a.steer_strength if a.steer else None, **metrics(cell)}
    pm = json.load(open(mq)); row["medmcqa_dev1000"] = round(pm["accuracy"] * 100, 2)
    # references
    for ref, key in ((a.ref_med, "med"), (a.ref_orig, "orig")):
        rp = ROOT / "results" / "cells" / ref
        if (rp / "labels_qwen3guard.jsonl").exists() and ref != cell:
            m = metrics(ref)
            for k in list(m):
                if k.startswith("asr|") or k.startswith("orr_"): row[f"d_{key}|{k}"] = round(row[k] - m[k], 2)
        if key == "med" and (rp / "medmcqa_dev1000.json").exists() and ref != cell:
            pr = json.load(open(rp / "medmcqa_dev1000.json"))["preds"]
            byid = {p["prompt_id"]: p["correct"] for p in pm["preds"]}
            ref_v = [p["correct"] for p in pr]; new_v = [byid[p["prompt_id"]] for p in pr]
            row["medmcqa_vs_med"] = {"diff_points": round((sum(new_v) - sum(ref_v)) / len(ref_v) * 100, 2), **mcnemar_from_preds(ref_v, new_v)}
    row["seconds"] = round(time.time() - t0)
    with open(a.sweep, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k in ("name", "asr|all", "asr|en", "asr|hi", "asr|hinglish", "orr_general|all", "orr_clinical|all", "medmcqa_dev1000", "medmcqa_vs_med", "seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
