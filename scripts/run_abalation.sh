# SEAD with frequency estimation on WikiMIA-32

for model in llama2-13b opt-6.7b falcon-7b pythia-6.9b; do
    python run_baselines.py \
        --gpu_ids 6 \
        --target_model $model \
        --data WikiMIA \
        --length 32 \
        --attack_method sead \
        --temperature 0.7 \
        --estimation_method frequency \
        --surrogate_model gpt2-xl
done
