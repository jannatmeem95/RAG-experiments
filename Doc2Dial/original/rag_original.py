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

# model = 'meta-llama/Llama-2-13b-chat-hf'

# device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
# print(device)

# tokenizer = AutoTokenizer.from_pretrained(model)
# # model.eval()
# print(f"Model loaded on {device}")


# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto", # if you have GPU
#     temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
#     max_new_tokens=128,  
#     repetition_penalty=1.1,  # without this output begins repeating
#     return_full_text=False
# )

print('model loaded2')

with open('doc2dial_dial_test.json','r') as f:
  doc2dial_test_questions = json.load(f)['dial_data']

with open('doc2dial_doc.json','r') as f:
    docs = json.load(f)["doc_data"]

c=0
query_doc = dict()
for d, v in doc2dial_test_questions.items():
    documents = docs[d]
    for topic, dialogs in tqdm(v.items()):
      for dial in dialogs:
        for t in range(len(dial['turns'])-1, 0, -1):
          if dial['turns'][t]['role'] == 'agent' and dial['turns'][t]["da"] == "respond_solution":
            query = dial['turns'][t-1]['utterance'] +'__'+ dial["dial_id"]
            dialog_history = ""
            # query_context = ""
            for r in range(t-1):
            #   query_context =  dial['turns'][r]['role'] +': ' + dial['turns'][r]['utterance'] + query_context
              dialog_history += dial['turns'][r]['role'] + ': ' + dial['turns'][r]['utterance'] + '\n'
            query_doc[query] = {'domain': d,'id': dial["doc_id"], 'gold response': dial['turns'][t]['utterance'], 'history': dialog_history} #, 'text': documents[dial["doc_id"]]}
            break

print(len(query_doc))
with open('queries_w_docs.json','w') as f:
    json.dump(query_doc, f, indent = 4)

# results = dict()
# for q in tqdm(questions):
#     history = questions[q]['history']
#     context = questions[q]['Top 10 chunks'][:5]
#     query = 'User Utterance: ' + q + '\nDialog History: ' + history 
#     prompt = "You are an agent. Your task is to respond to the latest User utterance. You are provided with the dialog history. You must respond as an agent only by using facts from the 5 documents provided below.\n"+query+'\nDocuments:\n'+'\n'.join(context)+'Agent Response: '
#     res_sorted = pipeline(prompt)
#     result_mpnet = res_sorted[0]["generated_text"]
#     # print(f'question: {q}: {result_mpnet}')
#     # break
#     results[q] = result_mpnet
# #     # print(result_mpnet)

# with open('/home/sshay004/workspaceMeem/Doc2Dial/llm-output/llama-2-doc2dial-outputs.json','w') as f:
#     json.dump(results, f, indent = 6)


