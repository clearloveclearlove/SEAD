import os
import torch
import json
import math
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from transformers import MambaForCausalLM
from sentence_transformers.util import dot_score
from eval import *


os.environ['HF_ENDPOINT'] = 'hf-mirror.com'

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data


def prepare_dataset(data, length):
    if data == "WikiMIA":
        original_dataset = load_dataset("data/WikiMIA", split=f"WikiMIA_length{length}")
        original_dataset = convert_huggingface_data_to_list_dic(original_dataset)

    elif data == "WikiMIA-paraphrased":
        original_dataset = load_dataset("data/WikiMIA_paraphrased_perturbed", split=f"WikiMIA_length{length}_paraphrased")
        original_dataset = convert_huggingface_data_to_list_dic(original_dataset)
    elif data == "WikiMIA-perturbed":
        original_dataset = load_dataset("data/WikiMIA_paraphrased_perturbed", split=f"WikiMIA_length{length}_perturbed")
        original_dataset = convert_huggingface_data_to_list_dic(original_dataset)
        
    elif data == "WikiMIA-24":
        original_dataset = load_dataset("data/WikiMIA-24", split=f"WikiMIA_length{length}")
        original_dataset = convert_huggingface_data_to_list_dic(original_dataset)

    elif data == "WikiMIA-gpt4-paraphrased":
        try:
            with open("data/gpt4_paraphrased_wikimia.json", "r", encoding="utf-8") as f:
                original_dataset = json.load(f)
        except FileNotFoundError:
            print("Error: GPT-4 paraphrased dataset file not found. Run the paraphrasing script first.")
            raise

    elif data == "WikiMIA-back-translated":
        try:
            with open("data/back_translated_wikimia.json", "r", encoding="utf-8") as f:
                original_dataset = json.load(f)
        except FileNotFoundError:
            print("Error: back-translated dataset file not found. Run the paraphrasing script first.")
            raise

    elif data == "WikiMIA-translated":
        try:
            with open("data/translated_wikimia.json", "r", encoding="utf-8") as f:
                original_dataset = json.load(f)
        except FileNotFoundError:
            print("Error: translated dataset file not found. Run the paraphrasing script first.")
            raise

    elif data == "zh-WikiMIA-translated":
        try:
            with open("data/zh-CN_translated_wikimia.json", "r", encoding="utf-8") as f:
                original_dataset = json.load(f)
        except FileNotFoundError:
            print("Error: translated dataset file not found. Run the paraphrasing script first.")
            raise
    
    elif "mimir" in data:
        # mimir-dm
        name = data.split('-')[1]
        original_non_member = []
        with open(f'data/MIMIR-ngram/non-members/{name}.jsonl', 'r') as f:
            for line in f:
                text = json.loads(line)
                original_non_member.append(" ".join(text.split()[:length]))
        
        original_member = []
        with open(f'data/MIMIR-ngram/members/{name}.jsonl', 'r') as f:
            for line in f:
                text = json.loads(line)
                original_member.append(" ".join(text.split()[:length]))
                
        original_member = original_member[:len(original_non_member)]
        testing_samples = 350
        size = min(testing_samples, len(original_non_member), len(original_member))
        original_dataset = [{"input":text, "label":1} for text in original_member[:size]] + [{"input":text, "label":0} for text in original_non_member[:size]]
        

    return original_dataset

def load_model(name1, name2, device):
    if not name1:
        model1 = None
        tokenizer1 = None
         
    elif "pythia-6.9b" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("pythia-6.9b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("pythia-6.9b")
    elif "pythia-12b" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("pythia-12b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("pythia-12b")
    elif "pythia-2.8b" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("pythia-2.8b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("pythia-2.8b")
    elif "pythia-1.4b" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("pythia-1.4b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("pythia-1.4b")
    elif "pythia-160m" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("pythia-160m", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("pythia-160m")
    elif "pythia-6.9b-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("pythia-6.9b-deduped", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("pythia-6.9b-deduped")
    elif "pythia-2.8b-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("pythia-2.8b-deduped", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("pythia-2.8b-deduped")
    elif "pythia-1.4b-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("pythia-1.4b-deduped", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("pythia-1.4b-deduped")
    elif "pythia-160m-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("pythia-160m-deduped", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("pythia-160m-deduped")
    elif "llama-13b" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("llama-13b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("llama-13b")
    elif "llama2-7b" == name1:
        model1 = LlamaForCausalLM.from_pretrained("Llama-2-7b-hf", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = LlamaTokenizer.from_pretrained("Llama-2-7b-hf")
    elif "llama-30b" == name1:
        model1 = LlamaForCausalLM.from_pretrained("llama-30b",device_map=device, return_dict=True, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = LlamaTokenizer.from_pretrained("llama-30b")
    elif "mamba-1.4b" == name1:
        model1 = MambaForCausalLM.from_pretrained("AI-ModelScope/mamba-1___4b-hf", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("AI-ModelScope/mamba-1___4b-hf")
    elif "neox-20b" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("gpt-neox-20b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("gpt-neox-20b")
    elif "falcon-7b" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("falcon-7b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("falcon-7b")
    elif "opt-6.7b" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("opt-6.7b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("opt-6.7b")
    elif "gpt2-xl" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("gpt2-xl")
     #Qwen2.5-7B', 'falcon-7B','Mistral-7B-v0.3   
    elif "qwen2.5-7b" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("Qwen2.5-7B", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("Qwen2.5-7B")
    elif "mistral-7b" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("Mistral-7B-v0.3", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("Mistral-7B-v0.3")       
    else:
        """
        You could modify the code here to make it compatible with other models
        """
        raise ValueError(f"Model {name1} is not currently supported!")

    if name2 == None:
        model2 = None
        tokenizer2 = None

    elif "pythia-6.9b" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("pythia-6.9b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("pythia-6.9b")
    elif "pythia-2.8b" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("pythia-2.8b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("pythia-2.8b")
    elif "pythia-1.4b" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("pythia-1.4b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("pythia-1.4b")
    elif "pythia-160m" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("pythia-160m", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("pythia-160m")
    elif "pythia-70m" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("pythia-70m", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("pythia-70m")
    elif "pythia-6.9b-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("pythia-6.9b-deduped", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("pythia-6.9b-deduped")
    elif "pythia-2.8b-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("pythia-2.8b-deduped", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("pythia-2.8b-deduped")
    elif "pythia-1.4b-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("pythia-1.4b-deduped", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("pythia-1.4b-deduped")
    elif "pythia-160m-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("pythia-160m-deduped", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("pythia-160m-deduped")
    elif "llama2-13b" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("llama-13b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("llama-13b")
    elif "llama2-7b" == name2:
        model2 = LlamaForCausalLM.from_pretrained("Llama-2-7b-hf", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = LlamaTokenizer.from_pretrained("Llama-2-7b-hf")
    elif "llama2-30b" == name2:
        model2 = LlamaForCausalLM.from_pretrained("llama-30b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = LlamaTokenizer.from_pretrained("llama-30b")
    elif "mamba" == name1:
        model2 = MambaForCausalLM.from_pretrained("mamba-1.4b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("mamba-1.4b")
    elif "falcon-7b" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("falcon-7b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("falcon-7b")
    elif "opt-6.7b" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("opt-6.7b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("opt-6.7b")
    elif "gpt2-xl" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("gpt2-xl", return_dict=True, device_map=None, torch_dtype=torch.float16)
        # target_device = torch.device(f"cuda:{0}")
        model2 = model2.to(device)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("gpt2-xl")
    elif 'qwen2-7b' == name2:
        model2 = AutoModelForCausalLM.from_pretrained("Qwen2-7B", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("Qwen2-7B")
    elif 'gemma-2-2b' == name2:
        model2 = AutoModelForCausalLM.from_pretrained("gemma-2-2b", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("gemma-2-2b")
    elif "qwen2.5-7b" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("Qwen2.5-7B", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("Qwen2.5-7B")
    elif "mistral-7b" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("Mistral-7B-v0.3", return_dict=True, device_map=device, torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("Mistral-7B-v0.3")     
        
    else:
        """
        You could modify the code here to make it compatible with other models
        """
        raise ValueError(f"Model {name2} is not currently supported!")

    return model1, model2, tokenizer1, tokenizer2



def inference(text, model, tokenizer):
    """Compute log-likelihood score for given text."""
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs[0]
    return -loss.item()

def calculatePerplexity(sentence, model, tokenizer, device):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()

def calculateTextSimilarity(model, embedding_model, tokenizer, text, decoding, device, temperature=0.7):

    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        
    input_ids = input_ids.to(model.device)
    
    sementic_similarity = []
    for i in range(1,input_ids.size(1)):
        input_ids_processed = input_ids[0][:i].unsqueeze(0)
        if decoding == "greedy":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), do_sample=False, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        elif decoding == "nuclear":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), do_sample=True, max_new_tokens=1, temperature=temperature,top_k=0, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        elif decoding == "contrastive":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), penalty_alpha=0.6, max_new_tokens=1, top_k=4, pad_token_id=tokenizer.eos_token_id)
        
        generated_embedding = embedding_model.encode(tokenizer.decode(generation[0][-1]))
        label_embedding = embedding_model.encode([tokenizer.decode(input_ids[0][i])]) 
        score = dot_score(label_embedding, generated_embedding)[0].item()
        if score <= 0:
            score = 1e-16
        sementic_similarity.append(math.log(score))
    
    return  sementic_similarity

def fitting(model, embedding_model, tokenizer, text, decoding):
    
    sementic_similarity = calculateTextSimilarity(model, embedding_model, tokenizer, text, decoding, device=model.device)
    p1, all_prob, p1_likelihood = calculatePerplexity(text, model, tokenizer, device=model.device)

    slope, intercept = np.polyfit(np.array(sementic_similarity), np.array(all_prob), 1)

    return slope, intercept
