"""Equivalence check: repair/resta.py arithmetic == SafeTune RESTAWrapper.apply on sampled tensors (DARE off).

    python repair/check_resta.py [--alpha 0.25] [--n 6]
"""
import argparse, json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from resta import Delta, _index, tensor_sha
from safetensors import safe_open
from safetune.core.optim.resta import RESTAWrapper, RESTAConfig

ap = argparse.ArgumentParser(); ap.add_argument("--alpha", type=float, default=0.25); ap.add_argument("--n", type=int, default=6)
ap.add_argument("--med", default="models/merged/med")
a = ap.parse_args()
D = Delta(); med = _index(a.med)
names = [n for n in D.names() if n.endswith(".weight") and ("proj" in n or "embed" in n)]
import random; random.Random(13).shuffle(names); names = names[: a.n]
aligned = {n: safe_open(D.g[n], "pt").get_tensor(n) for n in names}
base = {n: safe_open(D.b[n], "pt").get_tensor(n) for n in names}
ft = {n: safe_open(med[n], "pt").get_tensor(n) for n in names}
ref = RESTAWrapper(aligned, base, RESTAConfig(alpha=a.alpha)).apply(ft)
ok = True
for n in names:
    mine = (ft[n].float() + a.alpha * D.get(n)).half()
    same = tensor_sha(mine) == tensor_sha(ref[n].half())
    maxd = (mine.float() - ref[n].float()).abs().max().item()
    print(f"{n}: identical={same} max|diff|={maxd:.2e}"); ok &= same
print("RESTA_EQUIVALENT" if ok else "RESTA_MISMATCH")
