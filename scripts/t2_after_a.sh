#!/usr/bin/env bash
# Runs after chain A: RQ1/RQ2 tables -> (wait for back-translation) WildGuard on the hand-label set -> kappa -> chain B stage 1.
cd /c/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 HF_HOME="C:/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon/models/hf_cache" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/c/lexsi/.venv/Scripts/python.exe
while ! grep -q T2_CHAIN_A_DONE logs/t2_chain_a.log; do sleep 60; done
echo "STEP rq_tables $(date +%H:%M)"; $P harness/rq_tables.py --out results/drift > logs/t2_rq_tables.log 2>&1; echo "exit $?"
while ! grep -q "\[bt\] DONE" logs/t2_backtranslate.log; do sleep 60; done
echo "STEP wildguard $(date +%H:%M)"; $P harness/judge_wildguard.py > logs/t2_wildguard.log 2>&1; echo "exit $?"
echo "STEP kappa $(date +%H:%M)"; $P harness/kappa.py > logs/t2_kappa.log 2>&1; echo "exit $?"
echo "T2_AFTER_A_DONE $(date +%H:%M)"
bash scripts/t2_chain_b.sh stage1
