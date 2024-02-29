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

with open('/home/sshay004/workspaceMeem/Doc2Dial/queries_w_history_cosine_top5_docs.json','r') as f:
    questions = json.load(f)

results = dict()
for q in tqdm(questions):
    history = questions[q]['history']
    context = questions[q]['Top 10 chunks'][:5]
    query = 'User Utterance: ' + q + '\nDialog History: ' + history 
    prompt = "You are an agent. Your task is to respond to the latest User utterance. You are provided with the dialog history. You must respond as an agent only by using facts from the 5 documents provided below.\n"+query+'\nDocuments:\n'+'\n'.join(context)+'Agent Response: '
    res_sorted = pipeline(prompt)
    result_mpnet = res_sorted[0]["generated_text"]
    # print(f'question: {q}: {result_mpnet}')
    # break
    results[q] = result_mpnet
#     # print(result_mpnet)

with open('/home/sshay004/workspaceMeem/Doc2Dial/llm-output/llama-2-doc2dial-outputs.json','w') as f:
    json.dump(results, f, indent = 6)


# context = ["For Your Widow Or Widower \nThere are about five million widows and widowers receiving monthly Social Security benefits based on their deceased spouse's earnings record. And , for many of those survivors, particularly aged women, those benefits are keeping them out of poverty. Widows and widowers can receive : reduced benefits as early as age 60 or full benefits at full retirement age or older. benefits as early as age 50 if they're disabled AND their disability started before or within seven years of your death. benefits at any age , if they have not remarried , and if they take care of your child who is under age 16 or disabled and receives benefits on your record. If applying for disability benefits on a deceased worker s record , they can speed up the application process if they complete an Adult Disability Report and have it available at the time of their appointment. We use the same definition of disability for widows and widowers as we do for workers.",
#  "There are limits on how much survivors may earn while they receive benefits. Benefits for a widow, widower, or surviving divorced spouse may be affected by several additional factors : If your widow, widower, or surviving divorced spouse remarries before they reach age 60 age 50 if disabled , they cannot receive benefits as a surviving spouse while they're married. If your widow, widower, or surviving divorced spouse remarries after they reach age 60 age 50 if disabled , they will continue to qualify for benefits on your Social Security record. However , if their current spouse is a Social Security beneficiary , they may want to apply for spouse's benefits on their record. If that amount is more than the widow's or widower's benefit on your record , they will receive a combination of benefits that equals the higher amount. If your widow, widower, or surviving divorced spouse receives benefits on your record , they can switch to their own retirement benefit as early as age 62. This assumes they're eligible for retirement benefits and their retirement rate is higher than their rate as a widow, widower, or surviving divorced spouse. In many cases , a widow or widower can begin receiving one benefit at a reduced rate and then, at full retirement age, switch to the other benefit at an unreduced rate. If your widow, widower, or surviving divorced spouse will also receive a pension based on work not covered by Social Security, such as government or foreign work , their Social Security",
#  'an unreduced rate. If your widow, widower, or surviving divorced spouse will also receive a pension based on work not covered by Social Security, such as government or foreign work , their Social Security benefits as a survivor may be affected.',
#  'How Much Would Your Survivors Receive \nHow much your family could receive in benefits depends on your average lifetime earnings. The higher your earnings were , the higher their benefits would be. We calculate a basic amount as if you had reached full retirement age at the time you die. These are examples of monthly benefit payments : Widow or widower, full retirement age or older 100 percent of your benefit amount ; Widow or widower , age 60 to full retirement age 71 to 99 percent of your basic amount ; Disabled widow or widower , age 50 through 59 71 percent ; Widow or widower , any age, caring for a child under age 16 75 percent ; A child under age 18 19 if still in elementary or secondary school or disabled 75 percent ; and Your dependent parent , age 62 or older : One surviving parent 82 percent. Two surviving parents 75 percent to each parent. Percentages for a surviving divorced spouse would be the same as above. There may also be a special lump - sum death payment.',
#  "n\nBenefits Planner: Survivors | Planning For Your Survivors \nAs you plan for the future , you'll want to think about what your family would need if you should die now. Social Security can help your family if you have earned enough Social Security credits through your work. You can earn up to four credits each year. In 2019 , for example , you earn one credit for each $1,360 of wages or self - employment income. When you have earned $5,440 , you have earned your four credits for the year. The number of credits needed to provide benefits for your survivors depends on your age when you die. No one needs more than 40 credits 10 years of work to be eligible for any Social Security benefit. But , the younger a person is , the fewer credits they must have for family members to receive survivors benefits. Benefits can be paid to your children and your spouse who is caring for the children even if you don't have the required number of credits. They can get benefits if you have credit for one and one - half years of work 6 credits in the three years just before your death."]

#q = """
# User Utterance: "What would my widower receive in benefits?"\nDialog History:
# User: "I need help with planning my Social Security survivor benefits."
# Agent: "Do you need information about planning for your future Social Security survivor benefits?"
# User: "Yes."
# """


# prompt = "You are an agent. Your task is to respond to the latest User utterance. You are provided with the dialog history. You must respond using the facts from the 5 documents provided below.\n"+q+'\nDocuments:\n'+'\n'.join(context)+'Agent Response: '
# res_sorted = pipeline(prompt)
# result_mpnet = res_sorted[0]["generated_text"]
# print(result_mpnet)


#### Evaluation #########

"""
response = "Based on the information provided, as a widower, you would be eligible to receive survivor benefits based on your late spouse's earnings record. The amount of the benefit would depend on your spouse's earnings history and your age at the time of their passing. If you are currently 60 years old, you could receive reduced benefits as early as age 60, or full benefits at full retirement age or older. Additionally, if you are caring for a child under the age of 16, you could receive a higher benefit amount. It is important to note that there"
metric_sacrebleu = load_metric("sacrebleu")

ref = ["In the event of your passing, your widower would start receiving reduced benefits at age 60 or full benefits if they are at retirement age or older."]
predictions = [response]
print(len(predictions))
reference = [ref]
print(len(reference))
metric_sacrebleu.add_batch(predictions=predictions, references=reference)
    
final_score = metric_sacrebleu.compute()["score"]

print(final_score)
"""