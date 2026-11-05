#!/usr/bin/env bash
# Chain A (Thu 5 Nov, laptop): the five RQ1/RQ2/RQ5-reference test cells -> MedMCQA test-500 -> judge all
#   -> dev-pass baselines (orig, med, ctrl) + MedMCQA dev-1,000 -> judge -> Qwen3Guard on the hand-label answers.
# Resumable: rerun the script and every finished prompt / cell is skipped.
cd /c/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 HF_HOME="C:/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon/models/hf_cache" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/c/lexsi/.venv/Scripts/python.exe
TEST="aya_en_test aya_hi_test aya_hinglish_test clinicsafe_hi_en clinicsafe_hi_hi clinicsafe_hi_hinglish xstest_en_test xstest_hi_test xstest_hinglish_test"
MQT=Datasets/final/medmcqa_test_500/prompts.jsonl
MQD=Datasets/final/medmcqa_dev_1000/prompts.jsonl

cell () {  # cell model weights
  local c=$1 m=$2 w=$3 flag=""
  [ "$w" = nf4 ] && flag="--nf4"
  echo "STEP gen $c $(date +%H:%M)"
  $P harness/run_cell.py --cell $c --model $m --weights $w --sets $TEST > logs/t2_gen_$c.log 2>&1; echo "exit $?"
  if [ ! -f results/cells/$c/medmcqa_test500.json ]; then
    echo "STEP medmcqa $c $(date +%H:%M)"
    $P scripts/eval/medmcqa_score.py --model $m --set $MQT --out results/cells/$c/medmcqa_test500.json $flag > logs/t2_mq_$c.log 2>&1; echo "exit $?"
  fi
}

cell orig__fp16__f16__tpl-official CohereLabs/tiny-aya-global fp16
cell med__fp16__f16__tpl-official  models/merged/med          fp16
cell ctrl__fp16__f16__tpl-official models/merged/ctrl         fp16
cell orig__nf4__f16__tpl-official  CohereLabs/tiny-aya-global nf4
cell med__nf4__f16__tpl-official   models/merged/med          nf4

echo "STEP judge_test $(date +%H:%M)"
$P harness/judge.py --cell orig__fp16__f16__tpl-official --cell med__fp16__f16__tpl-official --cell ctrl__fp16__f16__tpl-official \
   --cell orig__nf4__f16__tpl-official --cell med__nf4__f16__tpl-official --cell hand_labels__orig__fp16 > logs/t2_judge_test.log 2>&1; echo "exit $?"

dev () {  # cell model
  local c=$1 m=$2
  echo "STEP devgen $c $(date +%H:%M)"
  $P harness/run_cell.py --cell $c --model $m --weights fp16 --ids Datasets/final/dev_pass_ids.jsonl > logs/t2_gen_$c.log 2>&1; echo "exit $?"
  if [ ! -f results/cells/$c/medmcqa_dev1000.json ]; then
    echo "STEP medmcqa_dev $c $(date +%H:%M)"
    $P scripts/eval/medmcqa_score.py --model $m --set $MQD --out results/cells/$c/medmcqa_dev1000.json > logs/t2_mq_$c.log 2>&1; echo "exit $?"
  fi
}
dev dev_pass__orig__fp16 CohereLabs/tiny-aya-global
dev dev_pass__med__fp16  models/merged/med
dev dev_pass__ctrl__fp16 models/merged/ctrl

echo "STEP judge_dev $(date +%H:%M)"
$P harness/judge.py --cell dev_pass__orig__fp16 --cell dev_pass__med__fp16 --cell dev_pass__ctrl__fp16 > logs/t2_judge_dev.log 2>&1; echo "exit $?"
echo "T2_CHAIN_A_DONE $(date +%H:%M)"
