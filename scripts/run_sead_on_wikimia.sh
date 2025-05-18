#!/usr/bin/env bash

# SEAD attack on WikiMIA-32

GPU=6
DATA=WikiMIA
LEN=32
METHOD=sead
NUM_SAMPLES=50
TEMP=0.7

for MODEL in llama2-13b opt-6.7b pythia-6.9b neox-20b llama2-30b; do
  echo "Running SEAD on $MODEL with WikiMIA..."
  python run_sead.py \
    --gpu_ids $GPU \
    --target_model $MODEL \
    --data $DATA \
    --length $LEN \
    --attack_method $METHOD \
    --estimation_method frequency \
    --num_samples $NUM_SAMPLES \
    --temperature $TEMP
  echo

done
