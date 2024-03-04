from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from typing import TYPE_CHECKING
import torch
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType
)

if TYPE_CHECKING:
    from arg_parser import ModelArgs
    from transformers import PreTrainedTokenizer, PreTrainedModel

def get_model_for_training(
    model_args: "ModelArgs",
    tokenizer:"PreTrainedTokenizer"
    ) -> "PreTrainedModel":
    
    if model_args.compute_dtype == "bf16":
        compute_dtpye = torch.bfloat16
    elif model_args.compute_dtype == "fp16":
        compute_dtpye = torch.float16
    
    if model_args.full_parameters_training:
        bnb_config = None
    elif model_args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtpye,
            bnb_4bit_quant_type="nf4" if model_args.quantization_type == "nf4" else "fp4",
            bnb_4bit_use_double_quant=True if model_args.use_double_quant else False
        )
    elif model_args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map = "auto"
    )
    
    if model_args.resize_embedding_layer:
        model.resize_token_embeddings(len(tokenizer))
    
    if not model_args.full_parameters_training:
        
        lora_config = LoraConfig(
            r=model_args.lora_rank,
            target_modules=model_args.lora_target,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True if model_args.use_gradient_checkpointing else False
        )
        
        model = get_peft_model(model, lora_config)
        
    model.print_trainable_parameters
        
    return model
    
    
    