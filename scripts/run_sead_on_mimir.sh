#!/usr/bin/env bash

# SEAD attack on various MIMIR datase(ngram=13) 

gpu=6
len=32
method=sead
num_samples=50
temp=0.7

datasets=("mimir-dm-13" "mimir-github-13" "mimir-hackernews-13" \
          "mimir-pile_cc-13" "mimir-wikipedia-13" "mimir-arxiv-13" "mimir-pubmed-13")
models=("pythia-1.4b" "pythia-2.8b" "pythia-6.9b" "pythia-12b")

for data in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo "Running SEAD on ${model} with dataset ${data}..."
    python run_sead.py \
      --gpu_ids ${gpu} \
      --target_model ${model} \
      --data ${data} \
      --length ${len} \
      --attack_method ${method} \
      --surrogate_model ${surrogate} \
      --num_samples ${num_samples} \
      --temperature ${temp} \
      --estimation_method frequency \
    echo
  done
done
