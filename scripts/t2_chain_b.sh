#!/usr/bin/env bash
# Chain B: circuits v2.  stage1 = token sets + 6 full EAP-IG runs + tables (~1.5 h)
#                        stage2 = 12 split-half runs + validation with the preamble (3 langs) + tables (~5 h)
#   bash scripts/t2_chain_b.sh stage1 | stage2 | all
cd /c/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 HF_HOME="C:/Users/SUSHOVIT/OneDrive/Desktop/Lexsi_Hackathon/models/hf_cache" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CIRCUITKIT_SKIP_MEM_GUARD=1
P=/c/lexsi/.venv/Scripts/python.exe
STAGE=${1:-all}
run_ck () {  # tag model lang [half]
  local tag=$1 model=$2 lang=$3 half=$4 extra="" suf=""
  [ -n "$half" ] && extra="--half $half" && suf="_half-$half"
  if [ -f results/circuits/v2/${tag}_${lang}${suf}_eap-ig_scores.json ]; then echo "skip $tag $lang $half"; return; fi
  echo "STEP ck $tag $lang $half $(date +%H:%M)"
  $P locate/run_circuitkit_v2.py --tag $tag --model $model --lang $lang $extra > logs/t2_ck_${tag}_${lang}${suf}.log 2>&1; echo "exit $?"
}
if [ "$STAGE" = stage1 ] || [ "$STAGE" = all ]; then
  echo "STEP token_sets $(date +%H:%M)"; $P locate/token_sets_v2.py > logs/t2_token_sets_v2.log 2>&1; echo "exit $?"
  for lang in en hi hinglish; do
    run_ck orig CohereLabs/tiny-aya-global $lang
    run_ck med models/merged/med $lang
  done
  echo "STEP tables $(date +%H:%M)"; $P locate/tensor_map_v2.py > logs/t2_tensor_map_v2.log 2>&1; echo "exit $?"
  echo "T2_CHAIN_B1_DONE $(date +%H:%M)"
fi
if [ "$STAGE" = stage2 ] || [ "$STAGE" = all ]; then
  for lang in en hi hinglish; do for h in a b; do
    run_ck orig CohereLabs/tiny-aya-global $lang $h
    run_ck med models/merged/med $lang $h
  done; done
  for lang in en hi hinglish; do
    echo "STEP validate $lang $(date +%H:%M)"; $P locate/validate_circuit.py --lang $lang > logs/t2_validate_$lang.log 2>&1; echo "exit $?"
  done
  echo "STEP tables $(date +%H:%M)"; $P locate/tensor_map_v2.py > logs/t2_tensor_map_v2.log 2>&1; echo "exit $?"
  echo "T2_CHAIN_B2_DONE $(date +%H:%M)"
fi
