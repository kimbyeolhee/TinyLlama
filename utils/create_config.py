import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig

def create_bnb_config():
    """
    load LLM in 4bit quantization. it can divide the used memory by 4.
    it makes the model to import on smaller GPU.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config

def create_peft_config(modules):
    """
    Create Parameter Efficient Fine-Tuning config
    :param modules: Names of the modules to apply LoRA
    """
    return LoraConfig(
        r=16, # the dimension of the low-rank matrices
        lora_alpha=64, # scaling factor for the weight matrices
        target_modules=modules, # modules to apply LoRA
        lora_dropout=0.1, # dropout probability of the LoRA layers
        bias="none", # set to all to train all bias parameters
        task_type="CAUSAL_LM"
    )