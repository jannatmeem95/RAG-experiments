import json
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import corpus_bleu

with open('/home/sshay004/workspaceMeem/Doc2Dial/Jan29/queries_w_history_orig_doc_top5_chunks.json','r') as f:
    questions = json.load(f)
references = [[v["gold response"]] for k,v in questions.items()] 
print(len(references))
print(references[0])

with open('/home/sshay004/workspaceMeem/Doc2Dial/Jan29/llm-output/llama-2-doc2dial-outputs.json', 'r') as f:
    res = json.load(f)
predictions = [v for k, v in res.items()]
print(len(predictions))

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
