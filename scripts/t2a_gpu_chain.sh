#!/usr/bin/env bash
# Task 2a GPU chain: dev drift probe -> judge -> stats -> hand-label answers -> ctrl_e2 -> merges
cd /c/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 HF_HOME="C:/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon/models/hf_cache" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/c/lexsi/.venv/Scripts/python.exe
SETS="aya_en_dev aya_hi_dev aya_hinglish_dev xstest_en_dev"
echo "STEP probe_orig $(date +%H:%M)"; $P harness/run_cell.py --cell probe_dev__orig__fp16 --model CohereLabs/tiny-aya-global --sets $SETS > logs/t2a_probe_orig.log 2>&1; echo "exit $?"
echo "STEP probe_ctrl $(date +%H:%M)"; $P harness/run_cell.py --cell probe_dev__ctrl__fp16 --model models/merged/ctrl --sets $SETS > logs/t2a_probe_ctrl.log 2>&1; echo "exit $?"
echo "STEP probe_med $(date +%H:%M)"; $P harness/run_cell.py --cell probe_dev__med__fp16 --model models/merged/med --sets $SETS > logs/t2a_probe_med.log 2>&1; echo "exit $?"
echo "STEP judge $(date +%H:%M)"; $P harness/judge.py --cell probe_dev__orig__fp16 --cell probe_dev__ctrl__fp16 --cell probe_dev__med__fp16 > logs/t2a_judge_probe.log 2>&1; echo "exit $?"
echo "STEP stats $(date +%H:%M)"; $P harness/stats.py --cells probe_dev__orig__fp16 probe_dev__ctrl__fp16 probe_dev__med__fp16 --ref probe_dev__orig__fp16 --out results/probe_dev/summary.json > logs/t2a_stats_probe.log 2>&1; echo "exit $?"
echo "STEP hand_labels $(date +%H:%M)"; $P harness/run_cell.py --cell hand_labels__orig__fp16 --model CohereLabs/tiny-aya-global --ids Datasets/final/hand_labels/sample_to_label.jsonl > logs/t2a_hand_labels.log 2>&1; echo "exit $?"
echo "STEP ctrl_e2 $(date +%H:%M)"; $P finetune/train_lora.py Datasets/final/sft_control/train.jsonl models/runs/ctrl_e2 --precision fp16 --grad_accum 16 --epochs 2 > logs/t2a_train_ctrl_e2.log 2>&1; echo "exit $?"
echo "STEP merges $(date +%H:%M)"; CUDA_VISIBLE_DEVICES="" $P finetune/merge.py models/runs/ctrl_e2/adapter models/merged/ctrl_e2 > logs/t2a_merge_ctrl_e2.log 2>&1; echo "exit $?"; CUDA_VISIBLE_DEVICES="" $P finetune/merge.py models/runs/med_e2/adapter models/merged/med_e2 > logs/t2a_merge_med_e2.log 2>&1; echo "exit $?"
echo "T2A_GPU_CHAIN_DONE $(date +%H:%M)"
