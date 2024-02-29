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

# embed_model_id = 'sentence-transformers/all-mpnet-base-v2'

# device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
# print(device)

# embed_model = HuggingFaceEmbeddings(
#     model_name=embed_model_id,
#     model_kwargs={'device': device},
#     encode_kwargs={'device': device, 'batch_size': 32}
# )

# print('embed model loaded')

model = 'meta-llama/Llama-2-13b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(device)

tokenizer = AutoTokenizer.from_pretrained(model)
# model.eval()
print(f"Model loaded on {device}")


# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     model_id,
#     use_auth_token=hf_auth
# )

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto", # if you have GPU
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=32,  
    repetition_penalty=1.1,  # without this output begins repeating
    return_full_text=False
)

print('model loaded2')

def readFile():

    with open('./questions_top10_all.json', 'r') as f:
        data = json.load(f)
    
    return data

def main():
    data = readFile()
    print('data loaded')

    t= open('./logs/llama-2-result-Jan15.txt','w') 


    results_mpnet = dict()

    h = 0

    for q, value in tqdm(data.items()):
        ctx_sorted = list(value["ctx"].values())[:5]

        context_sorted = ""
  
        i=1
        for p in ctx_sorted:
            context_sorted+= str(i)+'. '+p+'\n'
            i+=1
        
        options = ""
        op = ['1','2','3','4']
        for j in range(len(value["options"])):
            options = 'Option '+ str(j+1)+ '. '+value["options"][j]+'\n'

        prompt = 'Choose the correct answer for the following question : \nQuestion: ' + q + '\nfrom the following Options:\n'+ options+ '\nYou are provided with the 5 articles below and you must answer using the facts from the following articles.\nArticles:\n' + context_sorted + '\nAnswer:'
        res_sorted = pipeline(prompt)
        result_mpnet = res_sorted[0]["generated_text"]
        results_mpnet[q] =  result_mpnet

        if h ==0:
            print(len(ctx_sorted))
            print(options)
            # print(ctx_sorted)
            
        # if h>5:
        #     break
        # print(f'{q}: {result_mpnet}')
        h+=1

        g = {q: result_mpnet}
        with open('./logs/llama-2-result-Jan15.txt','a') as f:
            f.write(json.dumps(g))
            f.write('\n')

    out_dir = './pipeline-results/'

    with open(out_dir+'mpnet-out-Quality.json','w') as f:
        json.dump(results_mpnet,f, indent =4)

main()