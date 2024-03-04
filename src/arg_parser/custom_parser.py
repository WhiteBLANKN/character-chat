from transformers import(
    HfArgumentParser,
    TrainingArguments
)
from .model_args import ModelArgs
from .character_args import CharacterArgs
from typing import Tuple

def parser() -> Tuple[ModelArgs, CharacterArgs, TrainingArguments]:
    hf_parser = HfArgumentParser(
        [ModelArgs, CharacterArgs, TrainingArguments]
    )
    model_args, character_args, training_args, unknown_args = \
        hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    if unknown_args:
        print(f"These args can not be classified by HfArgumentParser.Please check if it's a misspell.")
        print(unknown_args)
        
    return model_args, character_args, training_args