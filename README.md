# SEAD: A Surrogate-free Label-Only Membership Inference Attack against Pre-trained LLMs with Semantic-Aware Density

**Abstract**

Membership inference attacks (MIAs) aim to determine whether specific data was used to train a model. While existing MIAs against pre-trained Large Language Models (LLMs) typically require access to complete logits (probabilities), such access is often unavailable in real-world deployments where only generated text is exposed. Current label-only MIAs rely on surrogate models to estimate token probabilities, but we identify fundamental limitations: high sensitivity to surrogate model selection and significant probability estimation errors. To address these challenges, we propose **SEAD (Semantic-Aware Density)**, a novel surrogate-free label-only MIA that directly estimates token probabilities through Monte Carlo sampling of the target model itself. This approach eliminates surrogate dependency and reduces probability estimation errors by an order of magnitude. Furthermore, we introduce a semantic-aware density mechanism that enhances attack effectiveness by accounting for both exact token matches and semantically equivalent alternatives, recognizing that LLMs may reveal memorized content via paraphrases. Extensive evaluations demonstrate that SEAD consistently outperforms existing label-only attacks and achieves results comparable to most logits-based methods.

## Method Overview

SEAD operates in three key stages:

1. **Monte Carlo Sampling of Token Probabilities**
   For each token position in the input sequence, SEAD queries the target model multiple times (e.g., 50 samples) to collect a distribution of next-token predictions. This direct sampling eliminates the need for surrogate models and provides a more accurate estimate of the model’s probability assignments.

2. **Semantic-Aware Density Computation**
   We compute a density score by aggregating the sampled probability masses, weighting each token by its semantic similarity to the ground-truth token. Specifically, we use an NLI model to measure entailment probabilities between the true token and each sampled token, so that semantically equivalent alternatives contribute to the overall density.

3. **Perplexity-Based Membership Inference**
   From the semantic-aware density values, SEAD derives a perplexity score that reflects how ‘surprised’ the model is by the input. Lower perplexity indicates stronger memorization. Finally, we perform membership inference by thresholding this score.

Together, Monte Carlo sampling and semantic-aware density enable SEAD to achieve high accuracy without any surrogate model, robustly capturing LLM memorization behaviors.


## Code Usage

This project has been tested under Python 3.9. All required dependencies are listed in requirements.txt.
Below is an example of running SEAD on the WikiMIA-32 dataset with 50 samples and a temperature of 0.7:

```bash
python main.py \
  --gpu_ids 0 \
  --target_model pythia-6.9b \
  --data WikiMIA \
  --length 32 \
  --attack_method sead \
  --num_samples 50 \
  --temperature 0.7 \
  --estimation_method nli 
```

For batch experiments, see the provided scripts in the `scripts/` folder, for example:

```bash
bash scripts/run_sead_wikimia.sh
```

This will sequentially launch SEAD on multiple target models (e.g., llama2-13b, opt-6.7b, pythia-6.9b, neox-20b, llama2-30b) using the specified sampling settings. 
