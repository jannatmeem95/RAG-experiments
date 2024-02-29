import pickle
import os
import json
from torch import cuda
from transformers import pipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
import transformers
import torch
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm 

from torch import cuda#, bfloat16
import transformers

pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")


def readFile():
    with open('./questions_top10_all.json', 'r') as f:
        data = json.load(f)
    
    return data

def main():
    data = readFile()
    print('data loaded')

    t= open('./logs/falcon-result-Jan15.txt','w') 


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
        res_sorted = pipe(prompt,temperature=0.0,  max_new_tokens=64, return_full_text=False)
        result_mpnet = res_sorted[0]["generated_text"]
        results_mpnet[q] =  result_mpnet

        if h ==0:
            print(len(ctx_sorted))
            print(options)
            # print(ctx_sorted)
            
        

        g = {q: result_mpnet}
        with open('./logs/falcon-result-Jan15.txt','a') as f:
            f.write(json.dumps(g))
            f.write('\n')
        if h>5:
            break
        # print(f'{q}: {result_mpnet}')
        h+=1

    out_dir = './pipeline-results/'
   
    with open(out_dir+'falcon-mpnet-out-Quality.json','w') as f:
        json.dump(results_mpnet,f, indent =4)

main()