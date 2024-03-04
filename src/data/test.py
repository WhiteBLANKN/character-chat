import json
from transformers import AutoTokenizer

def reformat_dataset(raw_dataset_path):
    
    with open(raw_dataset_path, "r", encoding='utf-8') as file:
        for lines in file:
            content = json.loads(lines)
            print(content)
            break
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
reformat_dataset("/Users/jayblankn/github/chat-character/dataset.jsonl")