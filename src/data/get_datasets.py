from typing import TYPE_CHECKING, Optional
from string import Template
from .ernie import ernie_dataset_generator
from .oai import openai_dataset_generator

if TYPE_CHECKING:
    from arg_parser import CharacterArgs

from ..utils import get_logger

logger = get_logger(__name__)
    
DEFAULT_PROMPT_TEMPLATE = """
你需要帮助我完成一个角色对话的构建，角色名字是$character。
根据原著中角色的背景、人物关系，喜好，常用语和故事情节进行写作。
这场对话聚焦的主题是<|TOPIC|>。
$character的第一句话必须是向user打招呼或个性化的自我介绍。
请务必按照以下格式返回多轮对话，对话总字数控制在$single_chat_length字：
{"role": "$character", "content": ""}
{"role": "user", "content": ""}
{"role": "$character", "content": ""}
{"role": "user", "content": ""}
{"role": "$character", "content": ""}
{"role": "user", "content": ""}
{"role": "$character", "content": ""}
{"role": "user", "content": ""}
{"role": "$character", "content": ""}
"""

KEYWORD_PROMPT = """
我需要完成人物角色$character的一些剧本创作，请提供50个角色相关的故事关键词。
要求：要尽可能贴合原著，并且50个关键词能尽可能展示出完整的角色经历，同时不能有重复。
注意：只需要提供关键词，我会通过这些关键词来进行创作，分别完成50个剧本内容创作。
返回格式如下，中间无需插入其他分类：
1.关键词1
2.关键词2
...
50.关键词50
"""
    
def generate_datasets(character_args: "CharacterArgs") -> Optional[str]:
    
    character = character_args.character
    single_chat_length = character_args.single_chat_length
    
    if character_args.use_off_the_shelf_datasets:
        dataset_path = character_args.datasets_path
        return dataset_path
    
    if not character_args.use_custom_prompt:
        prompt_template = DEFAULT_PROMPT_TEMPLATE
    else:
        with open(character_args.custom_prompt_path, 'r') as file:
            prompt_template = file.read()
            logger.warning("You have choose to use custom prompt.Make sure it can work well with generation funciton.")
            logger.warning(f"{prompt_template}")
            logger.warning(f"The dafault prompt looks like \n{DEFAULT_PROMPT_TEMPLATE}")
    
    conversation_prompt_template = Template(prompt_template)
    conversation_prompt = conversation_prompt_template.substitute(character=character, single_chat_length=single_chat_length)
    
    keyword_prompt_template = Template(KEYWORD_PROMPT)
    keyword_prompt = keyword_prompt_template.substitute(character=character)
    
    if character_args.openai_api_key:
        dataset_path = openai_dataset_generator(character_args, conversation_prompt, keyword_prompt)
    elif character_args.ernie_api_key:
        dataset_path = ernie_dataset_generator(character_args, conversation_prompt, keyword_prompt)

    return dataset_path