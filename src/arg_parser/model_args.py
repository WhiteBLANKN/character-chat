from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List, Literal
from .utils import _is_package_available, special_print

@dataclass
class ModelArgs:
    r"""Arguments for llm-models."""
    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "You can pass the path or id(check in ms or hf) to build a model."}
    )
    download_from_modelscope: Optional[bool] = field(
        default=False,
        metadata={"help": "You can choose from `huggingface` or `modelscope` to download your model."}
    )
    padding_side: Optional[str] = field(
        default="right",
        metadata={"help": "Recommend `right` for SFT Training."}
    )
    special_tokens: Optional[List[str]] = field(
        default=lambda: ["<|EOT|>"],
        metadata={"help": "For Generation Task. Will be add into training data."}
    )
    resize_embedding_layer: Optional[bool] = field(
        default=False,
        metadata={"help": "Recommend True for stable generation performance."}
    )
    load_in_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use 4 bit quantization in training."}
    )
    load_in_8bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use 8 bit quantization in training."}
    )
    full_parameters_training: Optional[bool] = field(
        default=False,
        metadata={"help": "Do not use quantization when training. Require much more compute resource."}
    )
    compute_dtype: Optional[Literal['fp16', 'bf16']] = field(
        default="bf16",
        metadata={"help": "If your GPU does not support bf16, use fp16 instead."}
    )
    quantization_type: Optional[Literal['fp4', 'nf4']] = field(
        default="nf4",
        metadata={"help": "nf4 has higher precision and larger numerical range"}
    )
    use_double_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "This flag is used for nested quantization where the quantization constants from the first quantization are quantized again."}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "Check the lora paper for more details."}
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "Check the lora paper for more details.Usually be two times of lora_rank."}
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "The dropout ratio of lora layers."}
    )
    lora_target: Optional[List[str]] = field(
        default=lambda: ['q_proj', 'v_proj'],
        metadata={"help": "Check the model config for compatibility."}
    )
    use_gradient_checkpointing: Optional[str] = field(
        default=True,
        metadata={"help": "Selectively storing intermediate activations and recomputing them as needed during the backward pass."}
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "Truncate the end of the sequence."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to save the cache files of downloaded model cache."}
    )

    def __post_init__(self):
        
        if self.download_from_modelscope:
            try:
                _is_package_available('modelscope')
            except ImportError:
                print("use `pip install modelscope` for model downloading.")
            
            special_print("Download model from modelscope")
            from modelscope import snapshot_download
            self.model_name_or_path = snapshot_download(self.model_name_or_path, cache_dir=self.cache_dir)
            
        if self.load_in_4bit + self.load_in_8bit + self.full_parameters_training != 1:
            raise ValueError("You can only set one of the `load_in_4bit`, `load_in_8bit`, `full_parameters_training` to be Ture.")
