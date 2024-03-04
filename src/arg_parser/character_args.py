from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CharacterArgs:
    r"""Arguments used to configure datasets"""
    character: str = field(
        metadata={"help": "The name of the character you would like to build."}
    )
    use_custom_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, make sure you have pass the prompt template args."}
    )
    custom_prompt_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of your custom prompt used to generate conversation data.Should be a .txt file."}
    )
    use_off_the_shelf_datasets: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, make sure to offer the dataset path."}
    )
    datasets_path: Optional[str] = field(
        default=None,
        metadata={"help": "The datasets used for training the chatbot.Only supporting .json file."}
    )
    ernie_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Get API Reference: https://cloud.baidu.com/"}
    )
    ernie_secret_key: Optional[str] = field(
        default=None,
        metadata={"help": "Get API Reference: https://cloud.baidu.com/"}
    )
    openai_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Get API Reference: https://platform.openai.com/"}
    )
    dataset_output_dir: Optional[str] = field(
        default='.',
        metadata={"help": "Where to store your generated datasets."}
    )
    dataset_length: Optional[int] = field(
        default=100,
        metadata={"help": "The numbers of conversation generated for model training."}
    )
    single_chat_length: Optional[int] = field(
        default=500,
        metadata={"help": "The certain length for prompting the llm to generate conversation in single request."}
    )
    
    def __post_init__(self):
        
        if self.use_custom_prompt and self.custom_prompt_path is None:
            raise ValueError("You set `use_custom_prompt` Ture, but didn't offer a path for your prompt!")
        
        if self.use_off_the_shelf_datasets and self.datasets_path is None:
            raise ValueError("You set `use_off_the_shelf_datasets` True, but didn't offer a path for your dataset.")
        
        if self.openai_api_key and (self.ernie_api_key or self.ernie_secret_key):
            raise ValueError("You can only choose one api for generation task.")
        elif self.openai_api_key is None and not all([self.ernie_api_key, self.ernie_secret_key]):
            raise ValueError("Both `ernie_api_key` and `ernie_secret_key` should be set for generation task.")
        
        
            