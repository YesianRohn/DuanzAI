import json
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import argparse

parser = argparse.ArgumentParser(description='Evaluate the answer command line argument')
parser.add_argument('--test', help='The LLM Answer to evaluate', choices=['glm', 'gpt'], default='gpt')
parser.add_argument('--type', help='FineTuning Type, Only GPT Tested', choices=['none', '5shot'], default='none')
args = parser.parse_args()

# Load data from JSON
with open('data/task_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def calculate_similarity(str1, str2):
    # Using SequenceMatcher to calculate text similarity ratio
    similarity_ratio = SequenceMatcher(None, str1, str2).ratio()
    return similarity_ratio

def calculate_fuzzy_similarity(str1, str2):
    # Using fuzzywuzzy's fuzz ratio to calculate similarity
    fuzzy_ratio = fuzz.ratio(str1, str2)/100
    return fuzzy_ratio

count_exact_match = 0
count_similar_match = 0
for item in data:
    if args.test == 'glm':
        ans = item['glm_0shot_punchline']
    elif args.test == 'gpt':
        if args.type == 'none':
            ans = item['gpt_0shot_punchline']
        else:
            ans = item['gpt_5shot_punchline']
    punchline = item['punchline']

    if ans == punchline:
        count_exact_match += 1
    else:
        similarity_ratio = calculate_similarity(ans, punchline)
        fuzzy_ratio = calculate_fuzzy_similarity(ans, punchline)
        count_similar_match += min(1, max(similarity_ratio, fuzzy_ratio))

total_items = len(data)
accuracy_exact_match = count_exact_match / total_items
accuracy_similar_match = (count_exact_match + count_similar_match) / total_items

print(f"Evaluate LLM: {args.test}")
print(f"Exact Match Accuracy: {accuracy_exact_match:.2f}")
print(f"Similar Match Accuracy: {accuracy_similar_match:.2f}")
