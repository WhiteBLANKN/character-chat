import json
import re
import os
import random
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional, List, Dict
from openai import OpenAI

if TYPE_CHECKING:
    from arg_parser import CharacterArgs

def openai_dataset_generator(
    character_args: "CharacterArgs",
    conversation_prompt: str,
    keyword_prompt: str
    ) -> Optional[str]:
    
    dir_path = character_args.dataset_output_dir
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    full_path = os.path.join(dir_path, "raw_dataset.jsonl")
    
    client = OpenAI()
    
    keywords = openai_keyword_generator(keyword_prompt, client)
    
    for i in tqdm(range(character_args.dataset_length)):
        try:
            content = conversation_prompt.replace('<|TOPIC|>', keywords[random.randint(0,50)])
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "user", "content": content}
                ],
                temperature=0.7,
                n=1,
                stop=None,
            )
            result = response.choices[0].message.content
            dialogs = post_process_response(result)
            
            with open(full_path, 'a', encoding="utf-8") as f:
                json.dump(dialogs, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"Error:{e} in generating conversation:{i}")
            continue
        
    return full_path
    
def openai_keyword_generator(
    keyword_prompt: str,
    client: "OpenAI"
    ) -> List:
    
    for i in range(5):

        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "user", "content": f"{keyword_prompt}"}
                ],
                temperature=0.7,
                n=1,
                stop=None,
            )
            result = response.choices[0].message.content
            keywords = []

            lines = result.split('\n')
            for line in lines:
                match = re.match(r'\d+\.\s*(\w+)', line)
                if match:
                    keyword = match.group(1)
                    keywords.append(keyword)
            
            return keywords
            
        except Exception as e:
            print(f"{e}")
            print(f"Fails to obtain the keyword list.Will try again.")
            continue
        
    raise RuntimeError("Fails to obtain Topics.Check Your Connection with OAI")

def post_process_response(result):
    
    pattern = re.compile(r'\{[^}]+\}', re.DOTALL)
    matches = pattern.findall(result)
    dialogs = [json.loads(json_str) for json_str in matches]
    
    return dialogs