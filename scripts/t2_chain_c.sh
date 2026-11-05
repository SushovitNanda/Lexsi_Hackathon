#!/usr/bin/env bash
# Chain C (repair sweeps, part 1): ReSta alpha x 6 dev passes -> steering calibration -> layer screen (4 layers)
#   -> circuit-restricted ReSta at alpha=0.25 x 3 tensor lists (needs locate/cresta_tensors_v2_top*.json from chain B).
# Every dev pass = 360 generations + judge + MedMCQA dev-1,000 (~15 min). Resumable (dev_pass.py skips finished cells).
#   bash scripts/t2_chain_c.sh
cd /c/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 HF_HOME="C:/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon/models/hf_cache" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/c/lexsi/.venv/Scripts/python.exe
MED=models/merged/med
dp () { echo "STEP devpass $1 $(date +%H:%M)"; $P harness/dev_pass.py --name "$@" >> logs/t2_chain_c_devpass.log 2>&1; echo "exit $?"; }
for a in 0.10 0.15 0.20 0.25 0.30 0.40; do dp resta_a$a --model $MED --resta $a; done
if [ ! -f models/steer/med_trilingual.pt ]; then
  echo "STEP steer_calibrate $(date +%H:%M)"; $P repair/steer_calibrate.py --model $MED --pairs trilingual --out models/steer/med_trilingual.pt > logs/t2_steer_calibrate.log 2>&1; echo "exit $?"
fi
# layer screen: 4 layers at one strength (strength 4 x unit-norm? no: raw CAA vector, scaled by STRENGTH)
STRENGTH=${STEER_STRENGTH:-1.0}
for L in 24 27 30 33; do dp steer_L${L}_s$STRENGTH --model $MED --steer models/steer/med_trilingual.pt --steer_layer $L --steer_strength $STRENGTH; done
for pct in 5 10 20; do
  [ -f locate/cresta_tensors_v2_top$pct.json ] && dp cresta_a0.25_top$pct --model $MED --resta 0.25 --resta_tensors locate/cresta_tensors_v2_top$pct.json
done
echo "STEP select $(date +%H:%M)"; $P harness/select.py > logs/t2_select.log 2>&1; echo "exit $?"
echo "T2_CHAIN_C_DONE $(date +%H:%M)"
