#!/usr/bin/env bash

# run_sead_on_mimir.sh
# SEAD attack on various MIMIR dataset 

GPU=6
LEN=32
METHOD=sead
NUM_SAMPLES=50
TEMP=0.7

datasets=("mimir-dm" "mimir-github" "mimir-hackernews" \
          "mimir-pile_cc" "mimir-wikipedia" "mimir-arxiv" "mimir-pubmed")
models=("pythia-1.4b" "pythia-2.8b" "pythia-6.9b" "pythia-12b")

for data in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo "Running SEAD on ${model} with dataset ${data}..."
    python run_sead.py \
      --gpu_ids ${GPU} \
      --target_model ${model} \
      --data ${data} \
      --length ${LEN} \
      --attack_method ${METHOD} \
      --num_samples ${NUM_SAMPLES} \
      --temperature ${TEMP} \
      --estimation_method nli \
      --nli_batch_size 64 \
    echo
  done
done
