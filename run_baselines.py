import os
import zlib
import numpy as np
import torch
import random
from pathlib import Path
from tqdm import tqdm
from options import Options
from utils import *
from eval import *
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from vectors import Sentence_Transformer
import torch.nn.functional as F

# Ensure NLTK tokenizers and taggers are available:
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


def ppl_attack(model, tokenizer, text, device):
    """Compute membership score as model perplexity."""
    perplexity, _, _ = calculatePerplexity(text, model, tokenizer, device)
    return perplexity


def reference_attack(model, tokenizer, ref_model, ref_tokenizer, text, device):
    """Compute score as difference in perplexity between reference and target models."""
    perp_target, _, _ = calculatePerplexity(text, model, tokenizer, device)
    perp_ref, _, _ = calculatePerplexity(text, ref_model, ref_tokenizer, device)
    return perp_ref - perp_target


def zlib_attack(model, tokenizer, text, device):
    """Compute score as perplexity divided by compressed size."""
    perp, _, _ = calculatePerplexity(text, model, tokenizer, device)
    comp_size = len(zlib.compress(text.encode('utf-8')))
    return perp / comp_size


def min_k_prob_attack(model, tokenizer, text, device, k=20):
    """Compute score using the lowest k% token log-probabilities."""
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    _, logits = outputs[:2]
    token_ids = input_ids[0][1:].unsqueeze(-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=token_ids).squeeze(-1)
    k_len = max(1, int(len(token_log_probs) * k / 100))
    lowest = np.sort(token_log_probs.cpu())[:k_len]
    return -np.mean(lowest).item()


def min_k_pp_attack(model, tokenizer, text, device, k=20):
    """Compute normalized lowest k% token scores."""
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    _, logits = outputs[:2]
    token_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=token_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = torch.clamp((probs * log_probs.pow(2)).sum(-1) - mu.pow(2), min=1e-6).sqrt()
    normed = (token_log_probs - mu) / sigma
    k_len = max(1, int(len(normed) * k / 100))
    lowest = np.sort(normed.cpu())[:k_len]
    return -np.mean(lowest).item()


def recall_attack(model, tokenizer, text, prefixes, device):
    """Compute recall score as conditional vs. unconditional log-likelihood."""
    # Unconditional log-likelihood
    perp, ll, _ = calculatePerplexity(text, model, tokenizer, device)
    unl = -ll
    if not prefixes:
        return 1.0
    prefix_text = " ".join(prefixes)
    prob, cond_ll, ppl, all_prob, loss = get_conditional_ll(prefix_text, text, model, tokenizer, device)
    recall_score = cond_ll / unl
    return -recall_score


def recall_attack_nli(model, tokenizer, text, prefixes, device, nli_model, nli_tokenizer):
    """Compute recall score using NLI-augmented sampling for conditional and unconditional scores."""
    # Unconditional score via sampling
    unl = semantic_sampling_attack(
        model, tokenizer, text,
        num_samples=50, temperature=0.7,
        top_k=None, top_p=None,
        estimation_method='nli',
        device=device,
        nli_model=nli_model,
        nli_tokenizer=nli_tokenizer
    )
    if not prefixes:
        return 1.0
    prefix_text = " ".join(prefixes)
    # Conditional over concatenated input+target
    concat_ids, prefix_len = None, None
    # Use semantic_sampling_attack with custom inputs
    cond = semantic_sampling_attack(
        model, tokenizer, text,
        num_samples=50, temperature=0.7,
        top_k=None, top_p=None,
        estimation_method='nli',
        device=device,
        nli_model=nli_model,
        nli_tokenizer=nli_tokenizer,
        _input_ids=concat_ids,
        prefix_len=prefix_len
    )
    return -(cond / unl)


def get_conditional_ll(input_text, target_text, model, tokenizer, device):
    """Compute conditional log-likelihood aligning with recall paper."""
    inp = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    tgt = tokenizer(target_text, return_tensors='pt').input_ids.to(device)
    concat = torch.cat([inp, tgt], dim=1)
    labels = concat.clone()
    labels[:, :inp.size(1)] = -100
    with torch.no_grad():
        outputs = model(concat, labels=labels)
    return get_all_prob(labels, outputs.loss, outputs.logits)


def get_all_prob(input_ids, loss, logits):
    """Extract per-token probabilities for conditional attack."""
    log_probs = F.log_softmax(logits, dim=-1)
    valid = (input_ids >= 0).nonzero(as_tuple=True)
    batch_idx, seq_idx = valid
    token_ids = input_ids[batch_idx, seq_idx]
    probs = []
    for b, s, tid in zip(batch_idx, seq_idx, token_ids):
        if s < logits.size(1):
            probs.append(log_probs[b, s, tid].item())
    ll = -loss.item()
    ppl = torch.exp(loss).item()
    prob = torch.exp(-loss).item()
    return prob, ll, ppl, probs, loss.item()


def neighborhood_attack(model, tokenizer, text, device, num_neighbors=10, neighbor_texts=None):
    """Compute neighborhood attack as difference between avg neighbor perplexity and original."""
    orig_perp, _, _ = calculatePerplexity(text, model, tokenizer, device)
    neigh_perps = [calculatePerplexity(n, model, tokenizer, device)[0] for n in neighbor_texts]
    return np.mean(neigh_perps) - orig_perp


def generate_neighbors(text, num_neighbors=10):
    """Generate simple random word-swapped neighbors."""
    words = word_tokenize(text)
    neighbors = []
    for _ in range(num_neighbors):
        neigh = words.copy()
        for _ in range(random.randint(1,3)):
            if len(neigh) <= 1: continue
            i, j = random.sample(range(len(neigh)), 2)
            neigh[i], neigh[j] = neigh[j], neigh[i]
        neighbors.append(" ".join(neigh))
    return neighbors


def petal_attack(model1, model2, embedding_model, tokenizer1, tokenizer2, text, decoding, temperature=0.7):
    """Compute PETAL attack score via semantic fitting and similarity."""
    slope, intercept = fitting(model2, embedding_model, tokenizer2, text, decoding=decoding)
    sims = calculateTextSimilarity(model1, embedding_model, tokenizer1, text,
                                   decoding=decoding, device=model1.device, temperature=temperature)
    estimates = [s * slope + intercept for s in sims]
    return -np.mean(estimates).item()


def word_substitution_attack(model, tokenizer, text, device):
    """Evaluate robustness via BERT-based word substitution."""
    perturbed = word_substitution_perturbation(text)
    return evaluate_robustness(model, tokenizer, text, perturbed, device)


def random_swap_attack(model, tokenizer, text, device):
    """Evaluate robustness via random adjacent word swaps."""
    perturbed = random_swap_perturbation(text)
    return evaluate_robustness(model, tokenizer, text, perturbed, device)


def back_translation_attack(model, tokenizer, text, device):
    """Evaluate robustness via back-translation perturbation."""
    perturbed = back_translation_perturbation(text)
    return evaluate_robustness(model, tokenizer, text, perturbed, device)


def word_substitution_perturbation(text, ratio=0.15):
    """Perform masked-word substitution using a fill-mask pipeline."""
    try:
        fill_mask = pipeline("fill-mask", model="bert-base-uncased")
    except:
        return simple_word_substitution(text, ratio)
    words = word_tokenize(text)
    tags = pos_tag(words)
    indices = [i for i, (_, t) in enumerate(tags) if t.startswith(('NN','VB','JJ'))]
    if not indices:
        return [text]
    perturbed = []
    for _ in range(3):
        wcopy = words.copy()
        for idx in random.sample(indices, max(1, int(len(indices)*ratio))):
            mask = wcopy.copy()
            mask[idx] = '[MASK]'
            masked_text = ' '.join(mask)
            try:
                preds = fill_mask(masked_text)
                for p in preds:
                    tok = p['token_str'].strip()
                    if tok.lower() != wcopy[idx].lower():
                        wcopy[idx] = tok
                        break
            except:
                pass
        perturbed.append(' '.join(wcopy))
    return perturbed


def simple_word_substitution(text, ratio=0.15):
    """Fallback: substitute from common synonyms dictionary."""
    common = {
        'good':['nice','great'], 'bad':['poor','awful'],
        'big':['large','huge'], 'small':['tiny','little']
    }
    words = word_tokenize(text)
    perturbed = []
    for _ in range(3):
        wcopy = words.copy()
        for idx in random.sample(range(len(words)), max(1,int(len(words)*ratio))):
            key = wcopy[idx].lower()
            if key in common:
                wcopy[idx] = random.choice(common[key])
        perturbed.append(' '.join(wcopy))
    return perturbed


def random_swap_perturbation(text, ratio=0.15):
    """Swap random adjacent tokens."""
    words = word_tokenize(text)
    perturbed = []
    for _ in range(3):
        wcopy = words.copy()
        for _ in range(max(1,int(len(words)*ratio))):
            i = random.randint(0,len(wcopy)-2)
            wcopy[i], wcopy[i+1] = wcopy[i+1], wcopy[i]
        perturbed.append(' '.join(wcopy))
    return perturbed


def back_translation_perturbation(text):
    """Perturb text by translating to German and back to English."""
    try:
        from transformers import MarianMTModel, MarianTokenizer
        en_de = 'Helsinki-NLP/opus-mt-en-de'
        de_en = 'Helsinki-NLP/opus-mt-de-en'
        tok_en_de = MarianTokenizer.from_pretrained(en_de)
        mdl_en_de = MarianMTModel.from_pretrained(en_de).to(torch.device('cpu'))
        tok_de_en = MarianTokenizer.from_pretrained(de_en)
        mdl_de_en = MarianMTModel.from_pretrained(de_en).to(torch.device('cpu'))
        mid = mdl_en_de.generate(**tok_en_de(text, return_tensors='pt', padding=True))
        de = tok_en_de.decode(mid[0], skip_special_tokens=True)
        back = mdl_de_en.generate(**tok_de_en(de, return_tensors='pt', padding=True))
        return [tok_de_en.decode(back[0], skip_special_tokens=True)]
    except:
        return word_substitution_perturbation(text)


def evaluate_robustness(model, tokenizer, text, perturbed_texts, device):
    """Compute difference in generation similarity as robustness score."""
    words = word_tokenize(text)
    prefix_len = max(1, int(len(words)*0.5))
    prefix = ' '.join(words[:prefix_len])
    ground = ' '.join(words[prefix_len:])
    inp_ids = torch.tensor(tokenizer.encode(prefix)).unsqueeze(0).to(device)
    with torch.no_grad():
        orig_out = model.generate(inp_ids, max_length=len(tokenizer.encode(text))+10)
    orig_dec = tokenizer.decode(orig_out[0], skip_special_tokens=True)[len(prefix):].strip()
    orig_sim = compute_text_similarity(orig_dec, ground)
    sims = []
    for pt in perturbed_texts:
        p_words = word_tokenize(pt)
        p_pref = ' '.join(p_words[:prefix_len])
        pid = torch.tensor(tokenizer.encode(p_pref)).unsqueeze(0).to(device)
        with torch.no_grad():
            pout = model.generate(pid, max_length=len(tokenizer.encode(pt))+10)
        pout_dec = tokenizer.decode(pout[0], skip_special_tokens=True)[len(p_pref):].strip()
        sims.append(compute_text_similarity(pout_dec, ground))
    return orig_sim - np.mean(sims)


def compute_text_similarity(text1, text2):
    """Compute Jaccard similarity between two texts."""
    a, b = set(word_tokenize(text1.lower())), set(word_tokenize(text2.lower()))
    if not a or not b:
        return 0.0
    return len(a&b) / len(a|b)


def process_prefix(model, tokenizer, prefixes, target_len, num_shots, pass_window=False):
    """Truncate prefixes to fit within model context window."""
    if pass_window:
        return prefixes[:num_shots]
    max_len = getattr(model.config, 'max_position_embeddings', 2048)
    counts = [len(tokenizer.encode(p)) for p in prefixes]
    cum = target_len
    allowed = 0
    for c in counts:
        if cum + c <= max_len:
            allowed += 1
            cum += c
        else:
            break
    return prefixes[:min(allowed, num_shots)]


def select_nonmember_prefixes(dataset, num_prefixes=12):
    """Select inputs labeled non-member as recall prefixes."""
    nonmem = [s for s in dataset if s['label']==0]
    if len(nonmem) < num_prefixes:
        print(f"Warning: only {len(nonmem)} non-member samples available")
        return [s['input'] for s in nonmem]
    return random.sample([s['input'] for s in nonmem], num_prefixes)


def evaluate_attacks(data, model1, model2, embed_model, tok1, tok2, args):
    """Run selected attack methods on the dataset and collect scores."""
    print(f"Dataset size: {len(data)}")
    print(f"Attack method: {args.attack_method}")
    prefixes = []
    if args.attack_method.lower() in ('all','recall'):
        prefixes = select_nonmember_prefixes(data, args.num_prefixes)
        avg_len = int(np.mean([len(tok1.encode(ex['input'])) for ex in data]))
        prefixes = process_prefix(model1, tok1, prefixes, avg_len, args.num_shots, args.pass_window)
        print(f"Using {len(prefixes)} prefixes for recall attack")
    neigh_data, num_neigh = None, None
    if args.attack_method.lower() in ('all','neighborhood'):
        neigh_data = prepare_dataset('WikiMIA-perturbed', args.length)
        num_neigh = len(neigh_data)//len(data)
    all_out = []
    for i, ex in enumerate(tqdm(data)):
        text = ex['input']
        scores = {}
        
        if args.attack_method.lower() in ('all','recall'):
            scores['RECALL'] = recall_attack(model1, tok1, text, prefixes, model1.device)
            
        if args.attack_method.lower() in ('all','ppl'):
            scores['PPL'] = ppl_attack(model1, tok1, text, model1.device)
            
        if args.attack_method.lower() in ('all','reference'):
            scores['REFERENCE'] = reference_attack(model1, tok1, model2, tok2, text, model1.device)
            
        if args.attack_method.lower() in ('all','zlib'):
            scores['ZLIB'] = zlib_attack(model1, tok1, text, model1.device)
            
        if args.attack_method.lower() in ('all','min-k'):
            scores['MIN-K%'] = min_k_prob_attack(model1, tok1, text, model1.device, k=args.min_k_percent)
            
        if args.attack_method.lower() in ('all','min-k++'):
            scores['MIN-K%++'] = min_k_pp_attack(model1, tok1, text, model1.device, k=args.min_k_percent)
            
        if args.attack_method.lower() in ('all','neighborhood'):
            neighs = [neigh_data[i*num_neigh+j]['input'] for j in range(num_neigh)]
            scores['NEIGHBORHOOD'] = neighborhood_attack(model1, tok1, text, model1.device, neighbor_texts=neighs)
            
        if args.attack_method.lower() in ('all','petal') and embed_model:
            scores['PETAL'] = petal_attack(model1, model2, embed_model, tok1, tok2, text, args.decoding)
            
        ex['pred'] = scores
        all_out.append(ex)
    return all_out


if __name__=='__main__':
    args = Options()
    parser = args.parser
    parser.add_argument('--attack_method', choices=["all","ppl","reference","zlib","min-k","min-k++","neighborhood","petal","recall"], default='all', help="Attack method to run")
    parser.add_argument('--min_k_percent', type=int, default=20, help="k% for Min-K attacks")
    parser.add_argument('--num_prefixes', type=int, default=12, help="Number of recall prefixes")
    parser.add_argument('--num_shots', type=int, default=5, help="Max recall shots")
    parser.add_argument('--pass_window', action='store_true', help="Ignore context length limits")
    args = parser.parse_args()

    os.environ['HF_ENDPOINT']='hf-mirror.com'
    os.environ['TOKENIZERS_PARALLELISM']='true'
    device = torch.device(f"cuda:{args.gpu_ids}")

    # Set output directory based on attack
    suffix = args.attack_method if args.attack_method!='all' else 'all'
    args.output_dir = f"{args.output_dir}/baselines_copy/{args.data}/{args.target_model}_{args.surrogate_model}_{suffix}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data and models
    data = prepare_dataset(args.data, args.length)
    m1, m2, t1, t2 = load_model(args.target_model, args.surrogate_model, device)
    embed_model = Sentence_Transformer(args.embedding_model, m1.device) if args.attack_method in ('all','petal') else None

    # Evaluate and plot
    results = evaluate_attacks(data, m1, m2, embed_model, t1, t2, args)
    fig_fpr_tpr(results, args.output_dir)
    print(f"Results saved to {args.output_dir}")
