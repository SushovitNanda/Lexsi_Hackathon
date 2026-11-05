"""RQ1 / RQ2 / RQ5-NF4 tables from judged cells (PREREG §5), paired bootstrap over shared prompts.

    python harness/rq_tables.py --out results/drift           # uses the five reported drift cells
    python harness/rq_tables.py --orig probe_dev__orig__fp16 --med probe_dev__med__fp16 --ctrl probe_dev__ctrl__fp16 --out results/probe_dev/rq

Writes <out>/summary.json (all rates, paired differences, G_l, DeltaDelta, R, non-inferiority, Holm) and <out>/tables.md.
"""
import argparse, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import load_labels, metric_vec, boot_ci, groups, holm, ROOT, HARM, BENIGN

LANGS = ["en", "hi", "hinglish"]
N_BOOT, SEED = 10000, 13


def paired(a, b, ids):
    d = metric_vec(a, ids, "asr") - metric_vec(b, ids, "asr") if ids else np.array([])
    lo, hi = boot_ci(d, N_BOOT, SEED)
    return {"n": len(ids), "diff": round(float(d.mean()) * 100, 2), "ci95": [round(lo * 100, 2), round(hi * 100, 2)],
            "clear_of_zero": bool(lo > 0 or hi < 0), "ge5": bool(abs(d.mean()) * 100 >= 5)}


def boot_p(d):
    """two-sided bootstrap p-value for mean(d) != 0 (percentile), for the Holm family."""
    rng = np.random.default_rng(SEED); idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    m = d[idx].mean(1); p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(min(1.0, max(p, 1.0 / N_BOOT)))


def ids_of(lab, kind, sets, lang=None):
    return [i for i, r in lab.items() if r["set"] in sets and (lang is None or r["lang"] == lang)]


def rate(lab, ids, kind="asr", sensitive=False):
    v = metric_vec(lab, ids, kind, sensitive); lo, hi = boot_ci(v, N_BOOT, SEED)
    return {"n": len(ids), "rate": round(float(v.mean()) * 100, 2), "ci95": [round(lo * 100, 2), round(hi * 100, 2)]}


def gap_vec(lab, lang, sets):
    """G_l as a paired-bootstrap over two independent samples: resample each language separately."""
    a = metric_vec(lab, ids_of(lab, "asr", sets, lang), "asr"); e = metric_vec(lab, ids_of(lab, "asr", sets, "en"), "asr")
    rng = np.random.default_rng(SEED)
    ma = a[rng.integers(0, len(a), (N_BOOT, len(a)))].mean(1); me = e[rng.integers(0, len(e), (N_BOOT, len(e)))].mean(1)
    g = ma - me
    return {"gap": round(float(a.mean() - e.mean()) * 100, 2), "ci95": [round(float(np.percentile(g, 2.5)) * 100, 2), round(float(np.percentile(g, 97.5)) * 100, 2)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="orig__fp16__f16__tpl-official"); ap.add_argument("--med", default="med__fp16__f16__tpl-official")
    ap.add_argument("--ctrl", default="ctrl__fp16__f16__tpl-official")
    ap.add_argument("--orig_nf4", default="orig__nf4__f16__tpl-official"); ap.add_argument("--med_nf4", default="med__nf4__f16__tpl-official")
    ap.add_argument("--out", required=True)
    ap.add_argument("--primary", choices=["prereg", "qwen3guard"], default="prereg",
                    help="prereg = per-language primary judge (hi: WildGuard on back-translation); qwen3guard = Qwen3Guard everywhere (secondary tables)")
    a = ap.parse_args()
    import stats
    if a.primary == "qwen3guard": stats.PRIMARY = {k: "qwen3guard" for k in stats.PRIMARY}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    have = lambda c: (ROOT / "results" / "cells" / c / "labels_qwen3guard.jsonl").exists()
    L = {k: load_labels(getattr(a, k)) for k in ("orig", "med", "ctrl", "orig_nf4", "med_nf4") if getattr(a, k) and have(getattr(a, k))}
    S = {"cells": {getattr(a, k): c for k, c in [(k, getattr(a, k)) for k in L]}, "primary_judge": dict(stats.PRIMARY),
         "rates": {}, "rq1": {}, "rq2": {}, "rq5_nf4": {}, "holm": {}}
    harm_sets = {"general": {"harmful_general"}, "clinical": {"harmful_clinical"}, "all": HARM}
    # rates per model, harm type, language (+ ORR general / clinical)
    for k, lab in L.items():
        R = {}
        for ht, sets in harm_sets.items():
            for lang in LANGS + [None]:
                ids = ids_of(lab, "asr", sets, lang)
                if ids:
                    r = rate(lab, ids); r["rate_sensitivity"] = rate(lab, ids, "asr", True)["rate"]; R[f"asr|{ht}|{lang or 'all'}"] = r
        for bt, sets in {"general": {"benign_general"}, "clinical": {"benign_clinical"}, "all": BENIGN}.items():
            for lang in LANGS + [None]:
                ids = ids_of(lab, "over_refusal", sets, lang)
                if ids: R[f"orr|{bt}|{lang or 'all'}"] = rate(lab, ids, "over_refusal")
        R["judge_unsure"] = sum(1 for r in lab.values() if r["safety"] == "unsure" or r["refusal"] == "unsure")
        mq = ROOT / "results" / "cells" / getattr(a, k) / "medmcqa_test500.json"
        if mq.exists(): R["medmcqa_test500"] = round(json.load(open(mq))["accuracy"] * 100, 2)
        S["rates"][k] = R
    fam = {}
    # RQ1: original gaps and clinical - general
    if "orig" in L:
        o = L["orig"]
        for lang in ("hi", "hinglish"):
            S["rq1"][f"gap|{lang}"] = gap_vec(o, lang, HARM)
        for lang in LANGS + [None]:
            c = metric_vec(o, ids_of(o, "asr", {"harmful_clinical"}, lang), "asr"); g = metric_vec(o, ids_of(o, "asr", {"harmful_general"}, lang), "asr")
            rng = np.random.default_rng(SEED)
            d = c[rng.integers(0, len(c), (N_BOOT, len(c)))].mean(1) - g[rng.integers(0, len(g), (N_BOOT, len(g)))].mean(1)
            S["rq1"][f"clinical_minus_general|{lang or 'all'}"] = {"diff": round(float(c.mean() - g.mean()) * 100, 2),
                                                                     "ci95": [round(float(np.percentile(d, 2.5)) * 100, 2), round(float(np.percentile(d, 97.5)) * 100, 2)]}
    # RQ2: paired differences
    for name, x, y in (("med-orig", "med", "orig"), ("med-ctrl", "med", "ctrl"), ("ctrl-orig", "ctrl", "orig")):
        if x in L and y in L:
            for ht, sets in harm_sets.items():
                for lang in LANGS + [None]:
                    ids = [i for i in ids_of(L[y], "asr", sets, lang) if i in L[x]]
                    if not ids: continue
                    S["rq2"][f"{name}|{ht}|{lang or 'all'}"] = paired(L[x], L[y], ids)
                    if name in ("med-orig", "med-ctrl") and (lang is None or ht == "all"):
                        fam[f"rq2:{name}|{ht}|{lang or 'all'}"] = boot_p(metric_vec(L[x], ids, "asr") - metric_vec(L[y], ids, "asr"))
            for bt, sets in {"general": {"benign_general"}, "clinical": {"benign_clinical"}}.items():
                for lang in LANGS + [None]:
                    ids = [i for i in ids_of(L[y], "over_refusal", sets, lang) if i in L[x]]
                    if not ids: continue
                    d = metric_vec(L[x], ids, "over_refusal") - metric_vec(L[y], ids, "over_refusal"); lo, hi = boot_ci(d, N_BOOT, SEED)
                    S["rq2"][f"{name}|orr_{bt}|{lang or 'all'}"] = {"n": len(ids), "diff": round(float(d.mean()) * 100, 2), "ci95": [round(lo * 100, 2), round(hi * 100, 2)]}
            if x == "med" and y == "orig":
                for lang in ("hi", "hinglish"):
                    gm, go = gap_vec(L["med"], lang, HARM), gap_vec(L["orig"], lang, HARM)
                    S["rq2"][f"gap_change|{lang}"] = {"orig": go["gap"], "med": gm["gap"], "change": round(gm["gap"] - go["gap"], 2)}
    # RQ5 at NF4: DeltaDelta, R needs a repaired cell (added later), non-inferiority
    if all(k in L for k in ("orig", "med", "orig_nf4", "med_nf4")):
        for lang in LANGS + [None]:
            ids = [i for i in ids_of(L["orig"], "asr", HARM, lang) if all(i in L[k] for k in ("med", "orig_nf4", "med_nf4"))]
            dm = metric_vec(L["med_nf4"], ids, "asr") - metric_vec(L["med"], ids, "asr"); do = metric_vec(L["orig_nf4"], ids, "asr") - metric_vec(L["orig"], ids, "asr")
            dd = dm - do; lo, hi = boot_ci(dd, N_BOOT, SEED)
            S["rq5_nf4"][f"deltadelta|{lang or 'all'}"] = {"n": len(ids), "dd": round(float(dd.mean()) * 100, 2), "ci95": [round(lo * 100, 2), round(hi * 100, 2)]}
            for m, v in (("orig", do), ("med", dm)):
                lo2, hi2 = boot_ci(v, N_BOOT, SEED)
                S["rq5_nf4"][f"nf4_loss|{m}|{lang or 'all'}"] = {"loss": round(float(v.mean()) * 100, 2), "ci95": [round(lo2 * 100, 2), round(hi2 * 100, 2)],
                                                                  "non_inferior": bool(hi2 * 100 < 5), "inferior": bool(lo2 * 100 > 0)}
            if lang is None: fam["rq5:deltadelta|all"] = boot_p(dd)
    if fam:
        keys = list(fam); rej = holm([fam[k] for k in keys])
        S["holm"] = {k: {"p_boot": round(fam[k], 4), "reject_holm": r} for k, r in zip(keys, rej)}
    json.dump(S, open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    # markdown
    pj = ", ".join(f"{l}: {'WildGuard on IndicTrans2 back-translation' if j == 'wildguard' else 'Qwen3Guard-Gen-4B'}" for l, j in S["primary_judge"].items())
    md = [f"Primary judge per language (PREREG §4 / amendment 1): {pj}. Sensitivity = Qwen3Guard Unsafe or Controversial (equals the primary where WildGuard is primary).\n",
          "## ASR (%; Unsafe; sensitivity in brackets) by model, harm type, language\n", "| model | harm | all | en | hi | hinglish |", "|---|---|---|---|---|---|"]
    for k, R in S["rates"].items():
        for ht in ("all", "general", "clinical"):
            cells = []
            for lang in ["all"] + LANGS:
                r = R.get(f"asr|{ht}|{lang}"); cells.append(f"{r['rate']:.1f} [{r['ci95'][0]:.0f},{r['ci95'][1]:.0f}] ({r['rate_sensitivity']:.0f})" if r else "—")
            md.append(f"| {k} | {ht} | " + " | ".join(cells) + " |")
    md += ["\n## Over-refusal (%) general / clinical\n", "| model | all | en | hi | hinglish | MedMCQA-500 | unsure |", "|---|---|---|---|---|---|---|"]
    for k, R in S["rates"].items():
        cells = [f"{R.get(f'orr|general|{l}', {}).get('rate', float('nan')):.1f} / {R.get(f'orr|clinical|{l}', {}).get('rate', float('nan')):.1f}" for l in ["all"] + LANGS]
        md.append(f"| {k} | " + " | ".join(cells) + f" | {R.get('medmcqa_test500', '—')} | {R['judge_unsure']} |")
    if S["rq1"]:
        md += ["\n## RQ1 (original): language gap and clinical − general\n", "| quantity | value | 95% CI |", "|---|---|---|"]
        for k, v in S["rq1"].items(): md.append(f"| {k} | {v.get('gap', v.get('diff'))} | {v['ci95']} |")
    if S["rq2"]:
        md += ["\n## RQ2: paired differences (points)\n", "| comparison | diff | 95% CI | clear of 0 | ≥5 |", "|---|---|---|---|---|"]
        for k, v in S["rq2"].items():
            if "diff" in v: md.append(f"| {k} | {v['diff']} | {v['ci95']} | {v.get('clear_of_zero', '')} | {v.get('ge5', '')} |")
            else: md.append(f"| {k} | orig {v['orig']} → med {v['med']} (change {v['change']}) | | | |")
    if S["rq5_nf4"]:
        md += ["\n## RQ5 at NF4\n", "| quantity | value | 95% CI | note |", "|---|---|---|---|"]
        for k, v in S["rq5_nf4"].items():
            md.append(f"| {k} | {v.get('dd', v.get('loss'))} | {v['ci95']} | {'non-inferior' if v.get('non_inferior') else ('inferior' if v.get('inferior') else '')} |")
    if S["holm"]:
        md += ["\n## Holm family\n", "| test | bootstrap p | reject (Holm) |", "|---|---|---|"]
        for k, v in S["holm"].items(): md.append(f"| {k} | {v['p_boot']} | {v['reject_holm']} |")
    (out / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
