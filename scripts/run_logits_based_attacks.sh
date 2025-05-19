#!/usr/bin/env bash

# run_logits_based_attacks.sh
# run white-box attacks on WikiMIA-32

GPU=6
DATA=WikiMIA
LEN=32
METHOD=white_box
SURROGATE=gpt2-xl  # Reference model for ref attack


MIN_K_PERCENT=20    # k% for Min-K and Min-K++
NUM_SHOTS=5         # Number of non-member samples for recall
NUM_PREFIXES=12     # Number of prefixes to select for recall
PASS_WINDOW=false   # Set to true to ignore context window limit (may OOM)

for MODEL in llama2-13b opt-6.7b pythia-6.9b neox-20b llama2-30b; do
  echo "Running white-box attacks on $MODEL with WikiMIA-32..."
  python main.py \
    --gpu_ids $GPU \
    --target_model $MODEL \
    --surrogate_model $SURROGATE \
    --data $DATA \
    --length $LEN \
    --attack_method $METHOD \
    --min_k_percent $MIN_K_PERCENT \
    --num_shots $NUM_SHOTS \
    --num_prefixes $NUM_PREFIXES \
    $( [ "$PASS_WINDOW" = true ] && echo "--pass_window" )
  echo

done
