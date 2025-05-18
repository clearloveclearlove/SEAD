import os
import numpy as np
import torch
import random
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from options import Options
from utils import *
from eval import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def set_seed(seed=42):
    """
    Set all random seeds to a fixed value and disable CUDA kernel nondeterminism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Apply deterministic settings
set_seed(42)


def load_nli_model(device, model_name="deberta-xlarge-mnli"):
    """
    Load a pretrained NLI model for computing entailment probabilities.

    Args:
        device: torch device to load the model onto
        model_name: HuggingFace model identifier

    Returns:
        model: sequence classification model
        tokenizer: corresponding tokenizer
    """
    try:
        print("Loading NLI model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading NLI model: {e}")
        return None, None


def compute_entailment_score(premise, hypothesis, nli_model, nli_tokenizer, device):
    """
    Compute the entailment classification between premise and hypothesis.

    Args:
        premise: premise text
        hypothesis: hypothesis text
        nli_model: loaded NLI model
        nli_tokenizer: tokenizer for NLI model
        device: torch device

    Returns:
        class_id: 2 for entailment, 1 for neutral, 0 for contradiction
    """
    if nli_model is None or nli_tokenizer is None:
        return 0
    try:
        inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = nli_model(**inputs)
            probs = torch.softmax(outputs.logits[0], dim=-1)
        # label mapping: 0=entailment,1=neutral,2=contradiction
        idx = torch.argmax(probs).item()
        # remap to desired class_id: entailment->2, neutral->1, contradiction->0
        mapping = {0: 2, 1: 1, 2: 0}
        return mapping[idx]
    except Exception as e:
        print(f"Error computing entailment score: {e}")
        return 0


def compute_entailment_scores_batch(premises, hypotheses, nli_model, nli_tokenizer, device, batch_size=16):
    """
    Batch compute entailment classes for pairs of texts.

    Args:
        premises: list of premise texts
        hypotheses: list of hypothesis texts
        nli_model: loaded NLI model
        nli_tokenizer: tokenizer for NLI model
        device: torch device
        batch_size: number of samples per batch

    Returns:
        list of class_ids: 2=entailment, 1=neutral, 0=contradiction
    """
    if nli_model is None or nli_tokenizer is None:
        return [0] * len(premises)
    if len(premises) != len(hypotheses):
        raise ValueError("Premises and hypotheses must have the same length")
    classes = []
    for i in range(0, len(premises), batch_size):
        batch_p = premises[i:i+batch_size]
        batch_h = hypotheses[i:i+batch_size]
        if len(batch_p) > 1:
            inputs = nli_tokenizer(batch_p, batch_h, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            with torch.no_grad():
                outputs = nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            idxs = torch.argmax(probs, dim=-1).cpu().tolist()
            mapping = {0: 2, 1: 1, 2: 0}
            classes.extend([mapping[idx] for idx in idxs])
        else:
            for p, h in zip(batch_p, batch_h):
                cid = compute_entailment_score(p, h, nli_model, nli_tokenizer, device)
                classes.append(cid)
    return classes


def semantic_sampling_attack(model, tokenizer, text, num_samples=10, temperature=0.3, top_k=50, top_p=0.95,
                             estimation_method="nli", device=None, nli_model=None, nli_tokenizer=None,
                             nli_batch_size=16):
    """
    Membership inference attack using sampling and optional NLI entailment.

    Args:
        model: target language model
        tokenizer: tokenizer for model
        text: input text
        num_samples: number of samples to generate per token
        temperature: sampling temperature
        top_k: top-k sampling parameter
        top_p: top-p sampling parameter
        estimation_method: 'frequency' or 'nli'
        device: torch device
        nli_model: NLI model for 'nli' estimation
        nli_tokenizer: tokenizer for NLI model
        nli_batch_size: batch size for NLI inference

    Returns:
        negative average token score
    """
    device = device or model.device
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    scores = []
    for i in range(1, input_ids.size(1)):
        context = input_ids[0][:i].unsqueeze(0)
        context_text = tokenizer.decode(context[0])
        actual_id = input_ids[0][i].item()
        actual_text = tokenizer.decode([actual_id])
        # Generate sample continuations
        gens = model.generate(context, max_new_tokens=1, do_sample=True, temperature=temperature,
                              top_k=top_k, top_p=top_p, num_return_sequences=num_samples,
                              pad_token_id=tokenizer.eos_token_id)
        gen_ids = [g[-1].item() for g in gens]
        counts = Counter(gen_ids)
        if estimation_method == "frequency":
            freq_prob = counts.get(actual_id, 0) / num_samples
            token_score = np.log(freq_prob + 1e-12) if num_samples > 1 else freq_prob
        elif estimation_method == "nli":
            unique_ids = list(counts.keys())
            freq_probs = {uid: cnt/num_samples for uid, cnt in counts.items()}
            premises = [actual_text]*len(unique_ids)
            hypotheses = [tokenizer.decode([uid]) for uid in unique_ids]
            ent_scores = compute_entailment_scores_batch(premises, hypotheses, nli_model, nli_tokenizer, device, batch_size=nli_batch_size)
            weighted = sum(freq_probs[uid]*es for uid, es in zip(unique_ids, ent_scores))
            token_score = np.log(weighted + 1e-12)
        else:
            raise ValueError(f"Unknown estimation method: {estimation_method}")
        scores.append(token_score)
    return -np.mean(scores)


def run_attack(model, tokenizer, text, ex, args, nli_model=None, nli_tokenizer=None):
    """Apply the semantic sampling attack to a single example."""
    ex["pred"] = {args.estimation_method: semantic_sampling_attack(
        model, tokenizer, text,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        estimation_method=args.estimation_method,
        device=model.device,
        nli_model=nli_model,
        nli_tokenizer=nli_tokenizer,
        nli_batch_size=args.nli_batch_size
    )}
    return ex


def evaluate_attack(data, model, tokenizer, args, nli_model=None, nli_tokenizer=None):
    """Evaluate the semantic sampling attack over the dataset."""
    print(f"Dataset size: {len(data)}")
    print(f"Running MIA attack with {args.num_samples} samples/token, temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print(f"Estimation method: {args.estimation_method}")
    if args.estimation_method == "nli": print(f"NLI batch size: {args.nli_batch_size}")
    results = []
    for ex in tqdm(data):
        text = ex["input"]
        results.append(run_attack(model, tokenizer, text, ex, args, nli_model, nli_tokenizer))
    return results


if __name__ == '__main__':
    opts = Options()
    parser = opts.parser
    parser.add_argument('--num_samples', type=int, default=20, help="Number of samples per token")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature")
    parser.add_argument('--top_k', type=int, default=None, help="Top-k sampling")
    parser.add_argument('--top_p', type=float, default=None, help="Top-p sampling")
    parser.add_argument('--estimation_method', choices=["frequency","nli"], default="nli", help="Probability estimation method")
    parser.add_argument('--nli_batch_size', type=int, default=64, help="Batch size for NLI inference")
    args = parser.parse_args()

    os.environ['HF_ENDPOINT'] = 'hf-mirror.com'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    device = torch.device(f"cuda:{args.gpu_ids}")

    # Set output directory
    args.output_dir = f"{args.output_dir}/mia_attack/{args.data}/{args.target_model}_{args.estimation_method}_samples{args.num_samples}_temp{args.temperature}_topk{args.top_k}_topp{args.top_p}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(args.data, args.length)
    model, _, tokenizer, _ = load_model(args.target_model, None, device)

    nli_model, nli_tokenizer = (None, None)
    if args.estimation_method == "nli":
        nli_model, nli_tokenizer = load_nli_model(device)

    outputs = evaluate_attack(dataset, model, tokenizer, args, nli_model, nli_tokenizer)
    fig_fpr_tpr(outputs, args.output_dir)
    print(f"Results saved to {args.output_dir}")
