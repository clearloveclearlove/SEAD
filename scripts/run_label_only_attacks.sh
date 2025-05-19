#!/usr/bin/env bash

# run_label_only_attacks.sh
# run PETAL attack on WikiMIA-32

GPU=6
DATA=WikiMIA
LEN=32
METHOD=petal
SURROGATE=gpt2-xl
EMBED_MODEL=all-MiniLM-L12-v2
DECODING=greedy

for MODEL in llama2-13b opt-6.7b pythia-6.9b neox-20b llama2-30b; do
  echo "Running PETAL attack on $MODEL with WikiMIA-32..."
  python main.py \
    --gpu_ids $GPU \
    --target_model $MODEL \
    --data $DATA \
    --length $LEN \
    --attack_method $METHOD \
    --surrogate_model $SURROGATE \
    --embedding_model $EMBED_MODEL \
    --decoding $DECODING
  echo

done
