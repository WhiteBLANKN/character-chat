import requests
import json
import re
import os
import random
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional, List, Dict

if TYPE_CHECKING:
    from arg_parser import CharacterArgs

def ernie_dataset_generator(
    character_args: "CharacterArgs",
    conversation_prompt: str,
    keyword_prompt: str
    ) -> Optional[str]:
    
    dir_path = character_args.dataset_output_dir
        
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
            
    full_path = os.path.join(dir_path, 'raw_dataset.jsonl')
    
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token="
    url += get_access_token(character_args)
    
    request_template = {
        "messages": [
            {
                "role": "user",
                "content": "content"
            }
        ]
    }
    headers = {
        'Content-Type': 'application/json'
    }
    
    keywords = ernie_keywords_generator(keyword_prompt, request_template, headers, url)
    
    for i in tqdm(range(character_args.dataset_length)):
        try:
            content = conversation_prompt.replace('<|TOPIC|>', keywords[random.randint(0,50)])
            request_template['messages'][0]['content'] = content
            payload = json.dumps(request_template)
            
            response = requests.request("POST", url, headers=headers, data=payload)
            
            dialogs = post_process_response(response.text)
            
            with open(full_path, 'a', encoding="utf-8") as f:
                json.dump(dialogs, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"Error:{e} in generating conversation:{i}")
            continue

    return full_path

def post_process_response(text):
    
    result_string = json.loads(text)["result"]
    
    pattern = re.compile(r'\{[^}]+\}', re.DOTALL)
    matches = pattern.findall(result_string)
    dialogs = [json.loads(json_str) for json_str in matches]
    
    return dialogs

def get_access_token(character_args: "CharacterArgs"):
        
    url = (
        "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id="
        f"{character_args.ernie_api_key}"
        f"&client_secret={character_args.ernie_secret_key}"
    )
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

def ernie_keywords_generator(
    keyword_prompt: str,
    request_template: str,
    headers: Dict,
    url:str
    ) -> List:

    request_template['messages'][0]['content'] = keyword_prompt
    payload = json.dumps(request_template)
    headers = {
        'Content-Type': 'application/json'
    }
    
    for i in range(5):
        if i <5:
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                break
            except Exception as e:
                print(f"Try to obtain topic fail in turn {i+1}.Will Try Again")
                print(f"{e}")
                continue
        else:
            raise RuntimeError(f"Fails to obtain Topics.Check Your Connection with {url}")
    
    result = json.loads(response.text)["result"]

    keywords = []

    lines = result.split('\n')
    for line in lines:
        match = re.match(r'\d+\.\s*(\w+)', line)
        if match:
            keyword = match.group(1)
            keywords.append(keyword)
    
    return keywords