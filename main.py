import os
import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

import zlib

from run_baselines import petal_attack, recall_attack, select_nonmember_prefixes, process_prefix, recall_attack_nli
from run_sead import semantic_sampling_attack, load_nli_model
from utils import load_model, prepare_dataset, inference
from vectors import Sentence_Transformer
from options import Options
from eval import compute_metrics


def run_attack_on_model(model_name, device_id, args):
    """Run selected attacks on a given model and dataset, then save results to CSV."""
    device = torch.device(f"cuda:{device_id}")

    # Load NLI model once
    nli_model, nli_tokenizer = load_nli_model(device)

    # Single dataset name
    dataset_name = args.data
    data = prepare_dataset(dataset_name, args.length)
    perturbed = prepare_dataset(f'{dataset_name}-perturbed', args.length)
    num_neighbors = len(perturbed) // len(data)

    # Load target and surrogate
    model, model2, tokenizer, tokenizer2 = load_model(
        model_name, args.surrogate_model, device
    )

    print(f"Running on model {model_name} ({dataset_name}) on device {model.device}")

    # Prepare recall prefixes
    prefixes = select_nonmember_prefixes(data, args.num_prefixes) if 'recall' in args.methods or 'recall_nli' in args.methods else []
    if prefixes:
        avg_len = int(np.mean([len(tokenizer.encode(ex['input'])) for ex in data]))
        prefixes = process_prefix(
            model, tokenizer, prefixes, avg_len, args.num_shots, args.pass_window
        )

    # Embedding model for PETAL
    embed_model = Sentence_Transformer(args.embedding_model, model.device) if 'petal' in args.methods else None

    scores = defaultdict(list)

    # Iterate examples
    for record in tqdm(data, desc=f"{model_name}-{dataset_name}"):
        text = record['input']
        label = record['label']

        # White-box methods
        if 'ppl' in args.methods or 'zlib' in args.methods or 'neighborhood' in args.methods or 'min-k' in args.methods or 'min-k++' in args.methods or 'reference' in args.methods:
            input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(model.device)
            outputs = model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]
            ppl = -torch.exp(loss).item()
            if 'ppl' in args.methods:
                scores['ppl'].append(ppl)
            if 'zlib' in args.methods:
                zscore = -loss.item() / len(zlib.compress(text.encode('utf-8')))
                scores['zlib'].append(zscore)

            # Neighborhood
            if 'neighborhood' in args.methods:
                neigh_scores = [inference(perturbed[i]['input'], model, tokenizer)
                                for i in range(num_neighbors)]
                diff = -loss.item() - np.mean(neigh_scores)
                scores['neighborhood'].append(diff)

            # Min-K and Min-K++
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)
            token_ids = torch.tensor(tokenizer.encode(text))[1:].unsqueeze(-1).to(model.device)
            token_log = log_probs.gather(dim=-1, index=token_ids).squeeze(-1)
            probs = torch.exp(log_probs)
            mu = (probs * log_probs).sum(-1)
            sigma = torch.clamp((probs * log_probs.pow(2)).sum(-1) - mu.pow(2), min=1e-6).sqrt()
            if 'min-k' in args.methods:
                mink = token_log.cpu().numpy()
                mink_score = np.mean(np.sort(mink)[: int(len(mink) * args.min_k_percent / 100)])
                scores['min-k'].append(mink_score)
            if 'min-k++' in args.methods:
                minkpp = ((token_log - mu) / sigma).cpu().numpy()
                minkpp_score = np.mean(np.sort(minkpp)[: int(len(minkpp) * args.min_k_percent / 100)])
                scores['min-k++'].append(minkpp_score)

            # Reference delta
            if 'reference' in args.methods and model2:
                loss2 = model2(
                    torch.tensor(tokenizer2.encode(text)).unsqueeze(0).to(model2.device),
                    labels=torch.tensor(tokenizer2.encode(text)).unsqueeze(0).to(model2.device)
                )[0]
                ref_delta = ppl - (-torch.exp(loss2).item())
                scores['reference'].append(ref_delta)

        # Recall-based
        if 'recall' in args.methods:
            rec_score = -recall_attack(model, tokenizer, text, prefixes, model.device)
            scores['recall'].append(rec_score)
        if 'recall_nli' in args.methods:
            rec_nli = -recall_attack_nli(
                model, tokenizer, text, prefixes,
                model.device, nli_model, nli_tokenizer
            )
            scores['recall_nli'].append(rec_nli)

        # PETAL
        if 'petal' in args.methods:
            petal_score = -petal_attack(
                model, model2, embed_model,
                tokenizer, tokenizer2,
                text, 'nuclear', temperature=args.temperature
            )
            scores['petal'].append(petal_score)

        # SEAD
        if 'sead' in args.methods:
            sead_score = -semantic_sampling_attack(
                model, tokenizer, text,
                num_samples=args.num_samples,
                temperature=args.temperature,
                nli_model=nli_model,
                nli_tokenizer=nli_tokenizer,
                nli_batch_size=args.nli_batch_size
            )
            scores['sead'].append(sead_score)

    # Aggregate metrics and save
    results = defaultdict(list)
    labels = [r['label'] for r in data]
    for method, vals in scores.items():
        auroc, fpr95, tpr01, tpr05, acc = compute_metrics(vals, labels)
        results['method'].append(method)
        results['auroc'].append(f"{auroc:.4f}")
        results['fpr95'].append(f"{fpr95:.4f}")
        results['tpr05'].append(f"{tpr05:.4f}")
        results['accuracy'].append(f"{acc:.4f}")

    df = pd.DataFrame(results)
    print(df)

    out_dir = Path(args.output_dir) / 'results' / f"{dataset_name}_{args.length}" / f"{model_name}_{args.surrogate_model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'results.csv', index=False)
    print(f"Saved results to {out_dir / 'results.csv'}")

    torch.cuda.empty_cache()
    return model_name


if __name__ == '__main__':
    opts = Options()
    parser = opts.parser

    # Add attack method and parameters
    parser.add_argument(
        '--attack_method', choices=['all', 'white_box', 'black_box'], default='all',
        help='Select attack category: white-box, black-box, or all'
    )
    
    # mink mink++
    parser.add_argument('--min_k_percent', type=int, default=20,
                        help='k% for Min-K and Min-K++')
    
    # RECALL
    parser.add_argument('--num_shots', type=int, default=5,
                        help='Number of non-member samples for recall')
    parser.add_argument('--num_prefixes', type=int, default=12,
                        help='Number of prefixes to select for recall')
    parser.add_argument('--pass_window', action='store_true',
                        help='Ignore context window for recall (may OOM)')
    
    
    # PETAL
    parser.add_argument('--target_model', type=str, default="pythia-160m", help="the model to attack")
    parser.add_argument('--surrogate_model', type=str, default="gpt2-xl")
    parser.add_argument('--embedding_model', type=str, default="all-MiniLM-L12-v2", help="the model used to compute semantic similarity")
    parser.add_argument('--decoding', type=str, default="greedy", help="the decoding strategy to use")
    
    # SEAD
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Samples per position for SEAD')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Top-p (nucleus) sampling')
    parser.add_argument('--estimation_method', choices=['frequency','nli'], default='nli',
                        help='Probability estimation method for SEAD')
    parser.add_argument('--nli_batch_size', type=int, default=64,
                        help='NLI batch size for SEAD and recall_nli')

    args = parser.parse_args()

    # Determine method set
    white = ['ppl','zlib','neighborhood','min-k','min-k++','reference','recall','recall_nli']
    black = ['petal','sead']
    if args.attack_method == 'white_box':
        args.methods = white
    elif args.attack_method == 'black_box':
        args.methods = black
    else:
        args.methods = white + black


    # Run
    run_attack_on_model(args.target_model, args.gpu_ids, args)