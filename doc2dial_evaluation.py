from datasets import load_metric
import json

with open('/home/sshay004/workspaceMeem/Doc2Dial/llm-output/llama-2-doc2dial-outputs.json','r') as f:
    responses = json.load(f)

with open('/home/sshay004/workspaceMeem/Doc2Dial/queries_w_gold_agent_response.json', 'r') as f:
    gold_responses = json.load(f)
references = [[v] for k,v in gold_responses.items()]

# print(len(references))
# # response = "Based on the information provided, as a widower, you would be eligible to receive survivor benefits based on your late spouse's earnings record. The amount of the benefit would depend on your spouse's earnings history and your age at the time of their passing. If you are currently 60 years old, you could receive reduced benefits as early as age 60, or full benefits at full retirement age or older. Additionally, if you are caring for a child under the age of 16, you could receive a higher benefit amount. It is important to note that there"
# metric_sacrebleu = load_metric("sacrebleu")

# # ref = ["In the event of your passing, your widower would start receiving reduced benefits at age 60 or full benefits if they are at retirement age or older."]
predictions = responses
# # print(len(predictions))
# # reference = [ref]
# # print(len(reference))
# metric_sacrebleu.add_batch(predictions=predictions, references=references)
    
# final_score = metric_sacrebleu.compute()["score"]

# print(final_score)


import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import corpus_bleu

# Example references and hypotheses for a batch
# references = [["The cat is on the mat.", "There is a cat on the mat."], ["The dog is in the garden."]]
# predictions = ["The cat is on the mat.", "The dog is in the garden."]

# Calculate BLEU scores for the entire batch
bleu_score4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25))
bleu_score1 = corpus_bleu(references, predictions, weights=(1,0,0,0))
bleu_score2 = corpus_bleu(references, predictions, weights=(0.5,0.5,0,0))
# Print the result
print(f"Corpus BLEU-4 Score: {bleu_score4}")
print(f"Corpus BLEU-1 Score: {bleu_score1}")
print(f"Corpus BLEU-2 Score: {bleu_score2}")
