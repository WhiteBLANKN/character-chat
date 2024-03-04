from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
    )
from peft import PeftModel
import torch
import configparser
import os

class SpecificSequenceCriteria(StoppingCriteria):
    def __init__(self, target_token):
        self.target_token = target_token
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        target_token_tensor = torch.tensor(self.target_token, device=input_ids.device)
        
        if not torch.equal(input_ids[0][-len(self.target_token):], target_token_tensor):
            return False
        else:
            return True
        
def get_output(prompt):
    model_input = tokenizer(prompt, return_tensors='pt')
    output = model.generate(input_ids=model_input['input_ids'].to('cuda'),
        max_new_tokens=1024,
        stopping_criteria=StoppingCriteriaList([specific_sequence_criteria]),
        repetition_penalty=1.1
        )

    return output, len(model_input['input_ids'][0])

def get_config():
    
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    
    return config

config = get_config()

output_dir = config.get('training_args', 'output_dir')
character = config.get('character_args', 'character')
model_name_or_path = config.get('model_args', 'model_name_or_path')
lora_adapter = os.path.join(output_dir, f'{character}')

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(output_dir, f'{character}/tokenizer')
    )

if config.get('model_args', 'full_parameters_training') == 'None':
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    
elif config.get('model_args', 'load_in_4bit') == 'True':
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if config.get('model_args', 'compute_dtype') == 'bf16' else torch.float16,
        bnb_4bit_quant_type='nf4' if config.get('model_args', 'quantization_type') == 'nf4' else 'fp4',
        bnb_4bit_use_double_quant=True if config.get('model_args', 'use_double_quant') == 'True' else False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        )
    
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model=model, model_id=lora_adapter)
elif config.get('model_args', 'load_in_8bit') == 'True':
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model=model, model_id=lora_adapter)
    
model.eval()

target_token = tokenizer.encode("<|END|>", add_special_tokens=False)
specific_sequence_criteria = SpecificSequenceCriteria(target_token=target_token)

def main():
    
    system_prompt = f"你是{character}，请保持{character}的语气，风格，性格等因素和我进行交流，交流内容如下：\n{character}:"
    prompt = system_prompt
    chat_length = 0
    print("\n\n" + '************************Start Chating************************')
    while chat_length <= 2048:
        output, previous_length = get_output(prompt)
        current_response = tokenizer.decode(output[0][previous_length:])
        print('\n' + f"{character}: " + current_response.split("<|END|>")[0].strip())
        USER_INPUT = input("\n你：")
        prompt = prompt + current_response + f"\n user: {USER_INPUT}" + f"\n{character}:"
        chat_length += len(output[0])
    
    print("达到对话长度上限")
        

if __name__ == '__main__':
    main()
