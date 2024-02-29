# from torch import cuda
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# embed_model_id = 'sentence-transformers/all-mpnet-base-v2'

# device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# embed_model = HuggingFaceEmbeddings(
#     model_name=embed_model_id,
#     model_kwargs={'device': device},
#     encode_kwargs={'device': device, 'batch_size': 32}
# )

import json

with open('/home/sshay004/workspaceMeem/NQ/unsupervised-passage-reranking/downloads/data/wikipedia-split/wiki_embeds/embeddings_sorted.json', 'r') as f:
    d = json.load(f)


# keys = list(d.keys())
# v= d[keys[0]]
# print(v.keys())
# answers_not_found = dict()
# not_found = 0
# for k,v in d.items():
#     a = False
#     for ans in v['answers']:
#         for text in v['texts']:
#             if ans in text:
#                 a=True
#                 break
#         if a == True:
#             break
#     if a == False:
#         not_found += 1
#         answers_not_found[k] = {'gold answers': v['answers'],'texts':v['texts']}

# print(not_found)
# with open('/home/sshay004/workspaceMeem/NQ/analysis/NQ_ans_not_in_top10.json','w') as f:
#     json.dump(answers_not_found,f,indent =6)

k = "where was the first season of slasher filmed"
v=d[k]
ctx = v['texts']

print(len(ctx))
print(ctx)