#!/usr/bin/env bash
# Hindi primary judge (PREREG amendment 1, 5 Nov): WildGuard on IndicTrans2 back-translation for every Hindi answer of
# the five reported test cells and the three dev-pass baselines, then the RQ1/RQ2/RQ5 tables under both primaries.
#   bash scripts/t2_hi_judge.sh            (~2 h GPU: back-translation ~3,100 Hindi items on the GPU, WildGuard 8-bit)
# Resumable (both scripts skip finished items). Run BEFORE chain C: dev_pass.py refuses to compute metrics on a
# reference cell whose Hindi answers lack WildGuard labels.
cd /c/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 HF_HOME="C:/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon/models/hf_cache" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/c/lexsi/.venv/Scripts/python.exe
SYS_PY="C:/Users/SUSHOVIT/python.exe"          # system Python 3.13: transformers 4.57 + IndicTransToolkit, CUDA
CELLS="orig__fp16__f16__tpl-official med__fp16__f16__tpl-official ctrl__fp16__f16__tpl-official orig__nf4__f16__tpl-official med__nf4__f16__tpl-official dev_pass__orig__fp16 dev_pass__med__fp16 dev_pass__ctrl__fp16"
ARGS=""; for c in $CELLS; do ARGS="$ARGS --cell $c"; done
echo "STEP backtranslate_hi $(date +%H:%M)"; IT_DEVICE=cuda "$SYS_PY" harness/backtranslate.py $ARGS --langs hi > logs/t2_bt_hi.log 2>&1; echo "exit $?"
echo "STEP wildguard_hi $(date +%H:%M)"; $P harness/judge_wildguard.py $ARGS --langs hi > logs/t2_wildguard_hi.log 2>&1; echo "exit $?"
echo "STEP rq_tables $(date +%H:%M)"; $P harness/rq_tables.py --out results/drift > logs/t2_rq_tables.log 2>&1; echo "exit $?"
echo "STEP rq_tables_qwen3guard $(date +%H:%M)"; $P harness/rq_tables.py --out results/drift_qwen3guard --primary qwen3guard > logs/t2_rq_tables_qwen3guard.log 2>&1; echo "exit $?"
echo "STEP dev_baseline_rows $(date +%H:%M)"
for m in orig med ctrl; do $P harness/dev_pass.py --name baseline_$m --model models/merged/$m --cell dev_pass__${m}__fp16 --sweep results/repair_dev/baselines_v2.jsonl >> logs/t2_dev_baseline_rows.log 2>&1; echo "exit $?"; done
echo "T2_HI_JUDGE_DONE $(date +%H:%M)"
