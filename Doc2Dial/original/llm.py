import pickle
import os
import json
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
import transformers
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm 

from torch import cuda#, bfloat16
import transformers
from datasets import load_metric

model = 'meta-llama/Llama-2-13b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(device)

tokenizer = AutoTokenizer.from_pretrained(model)
# model.eval()
print(f"Model loaded on {device}")


pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto", # if you have GPU
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=128,  
    repetition_penalty=1.1,  # without this output begins repeating
    return_full_text=False
)

print('model loaded2')


with open('doc2dial_doc.json','r') as f:
    docs = json.load(f)["doc_data"]

with open('queries_w_docs.json','r') as f:
    questions = json.load(f)

results = dict()
k = 0
for q in tqdm(questions):
    history = questions[q]['history']
    domain= questions[q]['domain']
    context = docs[domain][questions[q]['id']] 
    
    query = 'User Utterance: ' + q.split('__')[0] + '\nDialog History: ' + history 
    prompt = "You are an agent. Your task is to respond to the latest User utterance. You are provided with the dialog history: " + query + "\nGenerate an agent response using facts from the document provided below.\n"+'\nDocument:\n'+'\n'.join(context)+'Agent Response: '
    res_sorted = pipeline(prompt)
    result_mpnet = res_sorted[0]["generated_text"]
    if k < 5:
        ok = q.split('__')[0]
        print(f'question: {ok}: {result_mpnet}')
        k+=1
    # break
    results[q] = result_mpnet
#     # print(result_mpnet)

with open('/home/sshay004/workspaceMeem/Doc2Dial/llm-output/original_llama-2-doc2dial-outputs.json','w') as f:
    json.dump(results, f, indent = 6)