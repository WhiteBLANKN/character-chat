from datasets import load_dataset, Dataset
from typing import TYPE_CHECKING, Optional
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer
)
import json
import os
from tqdm import tqdm
from string import Template

if TYPE_CHECKING:
    from ..arg_parser import ModelArgs, CharacterArgs

CHAT_TEMPLATE = """
{% for message in messages %}
    {% if message['role'] == '$character' %}
{{ message['role'] }}: {{ message['content'] + " <|END|> " }}
    {% elif message['role'] == 'user' %}
{{ ' ' + message['role'] }}: {{ message['content'] }}
    {% endif %}
{% endfor %}
"""

def process_dataset_and_tokenizer(
        model_args:"ModelArgs",
        character_args: "CharacterArgs",
        raw_dataset_path: str) -> Optional[Dataset]:
    
    character = character_args.character

    #################
    ##GET TOKENIZER##
    #################
    template = Template(CHAT_TEMPLATE)
    chat_template = template.substitute(character=character)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    process_tokenizer(tokenizer, model_args, chat_template)

    ################
    ##GET DATASETS##
    ################
    dataset_path = reformat_dataset(raw_dataset_path, tokenizer, character)
    raw_dataset = load_dataset('json', data_files=dataset_path)
    column_names = raw_dataset["train"].column_names
    train_dataset = raw_dataset["train"].map(
        preprocess_func,
        fn_kwargs={"tokenizer": tokenizer, "max_length": model_args.max_length},
        remove_columns=column_names)
    #test_dataset = raw_dataset["test"].map

    return train_dataset, tokenizer


def preprocess_func(example, tokenizer: "PreTrainedTokenizer", max_length):
    prompt = tokenizer(example['prompt'])
    output = tokenizer(example['output'] + "</s>", add_special_tokens=False)

    input_ids = prompt["input_ids"] + output["input_ids"]
    attention_mask = prompt["attention_mask"] + output["attention_mask"]
    labels = [-100] * len(prompt["input_ids"]) + output["input_ids"]

    assert len(input_ids) == len(attention_mask) == len(labels)

    return {
        'input_ids': input_ids if len(input_ids) <= max_length else input_ids[:max_length],
        'attention_mask': attention_mask if len(attention_mask) <= max_length else attention_mask[:max_length],
        'labels': labels if len(labels) <= max_length else labels[:max_length]
    }


def process_tokenizer(
        tokenizer: "PreTrainedTokenizer",
        model_args: "ModelArgs",
        chat_template: str
        ) -> Optional[PreTrainedTokenizer]:

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = model_args.padding_side

    if model_args.resize_embedding_layer:
        tokenizer.add_tokens(model_args.special_tokens)

    tokenizer.chat_template = chat_template

def reformat_dataset(
        raw_dataset_path: str,
        tokenizer: "PreTrainedTokenizer",
        character: str,
        ) -> Optional[str]:
    
    system_prompt = f"你是{character}，请保持{character}的语气，风格，性格等因素和我进行交流，交流内容如下：\n"
    data_content = []
    total_lines = count_lines(raw_dataset_path)
    with open(raw_dataset_path, "r", encoding='utf-8') as file:
        for lines in tqdm(file, desc="Reformatting Dataset.", total=total_lines):
            try:
                content = json.loads(lines)
                conversation = tokenizer.apply_chat_template(content, tokenize=False)
                data_content.append({"prompt": system_prompt, "output": conversation})
            except Exception as e:
                print(f"ERROR:{e}\nThere is a error when formatting a generated data.")
                print("You can ignore it if the code continue runing.")
                continue
            
    output_dir_path, _ = os.path.split(raw_dataset_path)
    formatted_dataset_path = os.path.join(output_dir_path, "formatted_dataset.jsonl")
    with open(formatted_dataset_path, 'w', encoding="utf-8") as file:
        for i in tqdm(range(len(data_content))):
            if data_content[i]:
                json.dump(data_content[i], file, ensure_ascii=False)
                file.write("\n")

    return formatted_dataset_path

def count_lines(raw_dataset_path: str) -> Optional[int]:
    with open(raw_dataset_path, 'r', encoding="utf-8") as f:
        return sum(1 for _ in f)